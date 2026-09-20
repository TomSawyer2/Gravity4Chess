import type { AnalysisRequest, AnalysisResult } from '../types'
import type {
  EngineWorkerResponse,
  WorkerAnalyzeMessage,
} from './protocol'

interface PendingRequest {
  resolve: (result: AnalysisResult) => void
  reject: (error: Error) => void
}

class EngineClient {
  private worker: Worker | null = null
  private requestId = 0
  private pending = new Map<number, PendingRequest>()

  analyze(payload: AnalysisRequest): Promise<AnalysisResult> {
    this.ensureWorker()
    const requestId = ++this.requestId
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject })
      const message: WorkerAnalyzeMessage = { type: 'analyze', requestId, payload }
      this.worker?.postMessage(message)
    })
  }

  restart(): void {
    this.worker?.terminate()
    this.worker = null
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
    this.worker.onerror = (event) => {
      const error = new Error(event.message || '引擎 Worker 发生错误')
      for (const request of this.pending.values()) request.reject(error)
      this.pending.clear()
      this.worker?.terminate()
      this.worker = null
    }
  }
}

export const engineClient = new EngineClient()
