import type { AnalysisRequest, AnalysisResult, EngineError } from '../types'

export interface WorkerAnalyzeMessage {
  type: 'analyze'
  requestId: number
  payload: AnalysisRequest
}

export interface WorkerAnalyzeResponse {
  type: 'result'
  requestId: number
  payload: AnalysisResult | EngineError
}

export interface WorkerReadyResponse {
  type: 'ready'
  engineMode: 'wasm' | 'preview'
}

export type EngineWorkerResponse = WorkerAnalyzeResponse | WorkerReadyResponse
