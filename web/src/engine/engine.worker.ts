/// <reference lib="webworker" />

import type { AnalysisResult, EngineError } from '../types'
import type { WorkerAnalyzeMessage, WorkerAnalyzeResponse } from './protocol'
import { previewAnalyze } from './preview-engine'

interface WasmEngineModule {
  analyzePosition(
    movesCsv: string,
    maxDepth: number,
    timeLimitMs: number,
    tableMegabytes: number,
  ): string
}

let wasmModulePromise: Promise<WasmEngineModule | null> | null = null

async function loadWasmModule(): Promise<WasmEngineModule | null> {
  if (wasmModulePromise) return wasmModulePromise
  wasmModulePromise = (async () => {
    try {
      const base = `${self.location.origin}${import.meta.env.BASE_URL}`
      const moduleUrl = new URL('wasm/gravity4-engine.js', base).href
      const imported = (await import(/* @vite-ignore */ moduleUrl)) as {
        default: (options: {
          locateFile: (file: string) => string
        }) => Promise<WasmEngineModule>
      }
      return await imported.default({
        locateFile: (file) => new URL(file, moduleUrl).href,
      })
    } catch {
      return null
    }
  })()
  return wasmModulePromise
}

self.onmessage = async (event: MessageEvent<WorkerAnalyzeMessage>) => {
  if (event.data.type !== 'analyze') return
  const { requestId, payload } = event.data
  let result: AnalysisResult | EngineError

  try {
    const wasm = await loadWasmModule()
    if (wasm) {
      const raw = wasm.analyzePosition(
        payload.moves.join(','),
        payload.maxDepth,
        payload.timeLimitMs,
        payload.tableMegabytes,
      )
      const parsed = JSON.parse(raw) as Omit<AnalysisResult, 'engineMode'> | EngineError
      result = parsed.ok ? { ...parsed, engineMode: 'wasm' } : parsed
    } else {
      result = previewAnalyze(payload)
    }
  } catch (error) {
    result = {
      ok: false,
      error: error instanceof Error ? error.message : '未知引擎错误',
    }
  }

  const response: WorkerAnalyzeResponse = { type: 'result', requestId, payload: result }
  self.postMessage(response)
}

void loadWasmModule().then((module) => {
  self.postMessage({ type: 'ready', engineMode: module ? 'wasm' : 'preview' })
})
