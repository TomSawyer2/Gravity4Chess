import type { AnalysisRequest, AnalysisResult } from '../types'
import type {
  EngineWorkerResponse,
  WorkerAnalyzeMessage,
} from './protocol'
import { previewAnalyze } from './preview-engine'

interface PendingRequest {
  payload: AnalysisRequest
  resolve: (result: AnalysisResult) => void
  reject: (error: Error) => void
}

export class EngineClient {
  private worker: Worker | null = null
  private requestId = 0
  private pending = new Map<number, PendingRequest>()
  private previewFallback = false

  analyze(payload: AnalysisRequest): Promise<AnalysisResult> {
    if (this.previewFallback) return Promise.resolve(previewAnalyze(payload))
    this.ensureWorker()
    const requestId = ++this.requestId
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { payload, resolve, reject })
      const message: WorkerAnalyzeMessage = { type: 'analyze', requestId, payload }
      this.worker?.postMessage(message)
    })
  }

  restart(): void {
    this.worker?.terminate()
    this.worker = null
    this.previewFallback = false
    for (const request of this.pending.values()) {
      request.reject(new Error('搜索已取消'))
    }
    this.pending.clear()
  }

  private ensureWorker(): void {
    if (this.worker) return
    this.worker = new Worker(new URL('./engine.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (event: MessageEvent<EngineWorkerResponse>) => {
      if (event.data.type !== 'result') return
      const request = this.pending.get(event.data.requestId)
      if (!request) return
      this.pending.delete(event.data.requestId)
      if (!event.data.payload.ok) {
        request.reject(new Error(event.data.payload.error))
        return
      }
      request.resolve(event.data.payload)
    }
    const fallBackToPreview = () => {
      this.previewFallback = true
      for (const request of this.pending.values()) {
        try {
          request.resolve(previewAnalyze(request.payload))
        } catch (error) {
          request.reject(
            error instanceof Error ? error : new Error('预览引擎发生错误'),
          )
        }
      }
      this.pending.clear()
      this.worker?.terminate()
      this.worker = null
    }
    this.worker.onerror = (event) => {
      event.preventDefault()
      fallBackToPreview()
    }
    this.worker.onmessageerror = fallBackToPreview
  }
}

export const engineClient = new EngineClient()
