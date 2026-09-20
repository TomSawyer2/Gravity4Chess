import { create } from 'zustand'
import type {
  AnalysisResult,
  Difficulty,
  GameStatus,
  HistoryEntry,
  Side,
  ViewMode,
} from '../types'
import { DIFFICULTIES } from '../types'

interface GameState {
  moves: number[]
  history: HistoryEntry[]
  humanSide: Side
  status: GameStatus
  winner: Side | null
  currentAnalysis: AnalysisResult | null
  initialBlackWinRate: number
  reviewPly: number | null
  heatmapEnabled: boolean
  viewMode: ViewMode
  difficulty: Difficulty
  error: string | null
  begin: (humanSide: Side) => void
  setStatus: (status: GameStatus) => void
  setAnalysis: (analysis: AnalysisResult) => void
  recordMove: (move: number, analysis: AnalysisResult) => number[]
  replacePosition: (moves: number[], history: HistoryEntry[]) => void
  setReviewPly: (ply: number | null) => void
  setHeatmapEnabled: (enabled: boolean) => void
  setViewMode: (mode: ViewMode) => void
  setDifficulty: (difficulty: Difficulty) => void
  setError: (error: string | null) => void
}

export const useGameStore = create<GameState>()((set, get) => ({
  moves: [],
  history: [],
  humanSide: 'W',
  status: 'loading',
  winner: null,
  currentAnalysis: null,
  initialBlackWinRate: 0.5,
  reviewPly: null,
  heatmapEnabled: true,
  viewMode: 'play',
  difficulty: DIFFICULTIES[1],
  error: null,

  begin: (humanSide) => set({
    moves: [],
    history: [],
    humanSide,
    status: 'loading',
    winner: null,
    currentAnalysis: null,
    initialBlackWinRate: 0.5,
    reviewPly: null,
    viewMode: 'play',
    error: null,
  }),

  setStatus: (status) => set({ status }),

  setAnalysis: (analysis) => set((state) => ({
    currentAnalysis: analysis,
    winner: analysis.winner,
    status: analysis.terminal
      ? 'terminal'
      : analysis.sideToMove === state.humanSide
        ? 'human'
        : 'thinking',
    initialBlackWinRate: state.history.length === 0
      ? analysis.blackWinRate
      : state.initialBlackWinRate,
    error: null,
  })),

  recordMove: (move, analysis) => {
    const state = get()
    const candidate = analysis.candidates.find((item) => item.move === move)
    const entry: HistoryEntry = {
      ply: state.moves.length + 1,
      side: analysis.sideToMove,
      move,
      movesBefore: [...state.moves],
      analysis,
      blackWinRate: candidate?.blackWinRate ?? analysis.blackWinRate,
      score: candidate?.score ?? analysis.score,
    }
    const moves = [...state.moves, move]
    set({
      moves,
      history: [...state.history, entry],
      currentAnalysis: null,
      reviewPly: null,
      status: 'loading',
    })
    return moves
  },

  replacePosition: (moves, history) => set({
    moves,
    history,
    currentAnalysis: null,
    reviewPly: null,
    winner: null,
    status: 'loading',
    error: null,
  }),

  setReviewPly: (reviewPly) => set({
    reviewPly,
    viewMode: reviewPly === null ? 'play' : 'analysis',
  }),
  setHeatmapEnabled: (heatmapEnabled) => set({ heatmapEnabled }),
  setViewMode: (viewMode) => set({ viewMode }),
  setDifficulty: (difficulty) => set({ difficulty }),
  setError: (error) => set({ error, status: error ? 'error' : get().status }),
}))
