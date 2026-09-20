export type Side = 'B' | 'W'
export type EngineMode = 'wasm' | 'preview'
export type GameStatus = 'loading' | 'human' | 'thinking' | 'terminal' | 'error'
export type ViewMode = 'play' | 'analysis' | 'layers'

export interface CandidateAnalysis {
  move: number
  row: number
  col: number
  layer: number
  score: number
  sideWinRate: number
  blackWinRate: number
  tag: 'win' | 'block' | null
}

export interface SearchStats {
  depth: number
  elapsedMs: number
  nodes: number
  nps: number
  ttHitRate: number
}

export interface AnalysisResult {
  ok: true
  engineVersion: number
  engineMode: EngineMode
  ply: number
  sideToMove: Side
  heights: number[]
  stacks: Side[][]
  terminal: boolean
  winner: Side | null
  bestMove: number
  score: number
  blackWinRate: number
  candidates: CandidateAnalysis[]
  stats: SearchStats
}

export interface EngineError {
  ok: false
  error: string
}

export interface AnalysisRequest {
  moves: number[]
  maxDepth: number
  timeLimitMs: number
  tableMegabytes: number
}

export interface HistoryEntry {
  ply: number
  side: Side
  move: number
  movesBefore: number[]
  analysis: AnalysisResult
  blackWinRate: number
  score: number
}

export interface Difficulty {
  id: 'quick' | 'standard' | 'deep'
  label: string
  caption: string
  maxDepth: number
  timeLimitMs: number
}

export const DIFFICULTIES: Difficulty[] = [
  {
    id: 'quick',
    label: '快棋',
    caption: '每步约 120 ms',
    maxDepth: 20,
    timeLimitMs: 120,
  },
  {
    id: 'standard',
    label: '标准',
    caption: '每步约 400 ms',
    maxDepth: 24,
    timeLimitMs: 400,
  },
  {
    id: 'deep',
    label: '深思',
    caption: '固定深度 9',
    maxDepth: 9,
    timeLimitMs: 0,
  },
]

export function opposite(side: Side): Side {
  return side === 'B' ? 'W' : 'B'
}

export function sideLabel(side: Side): string {
  return side === 'B' ? '黑方' : '白方'
}

export function moveLabel(move: number): string {
  const row = Math.floor(move / 5)
  const col = move % 5
  return `${String.fromCharCode(65 + col)}${row + 1}`
}
