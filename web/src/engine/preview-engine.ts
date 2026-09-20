import type {
  AnalysisRequest,
  AnalysisResult,
  CandidateAnalysis,
  Side,
} from '../types'
import { opposite } from '../types'

const BOARD_SIZE = 5
const COLUMN_COUNT = 25
const LAYERS = 5
const DIRECTIONS = [
  [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, -1, 0],
  [0, 0, 1], [1, 0, 1], [-1, 0, 1], [0, 1, 1],
  [0, -1, 1], [1, 1, 1], [1, -1, 1], [-1, 1, 1],
  [-1, -1, 1],
] as const

function positionOf(move: number, layer: number): number {
  return move * LAYERS + layer
}

const winningLines: number[][] = []
for (let row = 0; row < BOARD_SIZE; row += 1) {
  for (let col = 0; col < BOARD_SIZE; col += 1) {
    for (let layer = 0; layer < LAYERS; layer += 1) {
      for (const [dr, dc, dl] of DIRECTIONS) {
        const line: number[] = []
        for (let step = 0; step < 4; step += 1) {
          const r = row + dr * step
          const c = col + dc * step
          const l = layer + dl * step
          if (r < 0 || r >= 5 || c < 0 || c >= 5 || l < 0 || l >= 5) {
            line.length = 0
            break
          }
          line.push(positionOf(r * 5 + c, l))
        }
        if (line.length === 4) winningLines.push(line)
      }
    }
  }
}

interface PreviewBoard {
  cells: Array<Side | null>
  stacks: Side[][]
  heights: number[]
  sideToMove: Side
  winner: Side | null
}

function buildBoard(moves: number[]): PreviewBoard {
  const cells = Array<Side | null>(125).fill(null)
  const stacks = Array.from({ length: COLUMN_COUNT }, () => [] as Side[])
  const heights = Array<number>(COLUMN_COUNT).fill(0)
  let side: Side = 'B'
  let winner: Side | null = null

  for (const move of moves) {
    if (!Number.isInteger(move) || move < 0 || move >= COLUMN_COUNT) {
      throw new Error('落子坐标无效')
    }
    if (winner) throw new Error('胜负已定后仍包含额外落子')
    const layer = heights[move]
    if (layer >= LAYERS) throw new Error('落子列已满')
    cells[positionOf(move, layer)] = side
    stacks[move].push(side)
    heights[move] += 1
    if (hasWin(cells, side)) winner = side
    side = opposite(side)
  }

  return { cells, stacks, heights, sideToMove: side, winner }
}

function hasWin(cells: Array<Side | null>, side: Side): boolean {
  return winningLines.some((line) => line.every((position) => cells[position] === side))
}

function wouldWin(board: PreviewBoard, side: Side, move: number): boolean {
  if (board.heights[move] >= LAYERS) return false
  const position = positionOf(move, board.heights[move])
  board.cells[position] = side
  const won = hasWin(board.cells, side)
  board.cells[position] = null
  return won
}

function evaluate(cells: Array<Side | null>): number {
  const weights = [0, 10, 200, 5000, 9_999_999]
  let score = 0
  for (const line of winningLines) {
    let black = 0
    let white = 0
    for (const position of line) {
      if (cells[position] === 'B') black += 1
      if (cells[position] === 'W') white += 1
    }
    if (black && white) continue
    if (black) score += weights[black]
    if (white) score -= weights[white]
  }
  return score
}

function probability(score: number): number {
  if (score > 9_000_000) return 1
  if (score < -9_000_000) return 0
  return 1 / (1 + Math.exp(-score / 6000))
}

export function previewAnalyze(request: AnalysisRequest): AnalysisResult {
  const startedAt = performance.now()
  const board = buildBoard(request.moves)
  const terminal = board.winner !== null || request.moves.length === 125
  if (terminal) {
    return {
      ok: true,
      engineVersion: 0,
      engineMode: 'preview',
      ply: request.moves.length,
      sideToMove: board.sideToMove,
      heights: board.heights,
      stacks: board.stacks,
      terminal: true,
      winner: board.winner,
      bestMove: -1,
      score: 0,
      blackWinRate: board.winner === 'B' ? 1 : board.winner === 'W' ? 0 : 0.5,
      candidates: [],
      stats: { depth: 0, elapsedMs: 0, nodes: 0, nps: 0, ttHitRate: 0 },
    }
  }

  const candidates: CandidateAnalysis[] = []
  const side = board.sideToMove
  for (let move = 0; move < COLUMN_COUNT; move += 1) {
    if (board.heights[move] >= LAYERS) continue
    const wins = wouldWin(board, side, move)
    const blocks = !wins && wouldWin(board, opposite(side), move)
    const layer = board.heights[move]
    const position = positionOf(move, layer)
    board.cells[position] = side
    let blackScore = evaluate(board.cells)
    board.cells[position] = null
    if (wins) blackScore = side === 'B' ? 99_999_998 : -99_999_998
    const sideScore = side === 'B' ? blackScore : -blackScore
    const centerDistance = Math.abs(Math.floor(move / 5) - 2) + Math.abs((move % 5) - 2)
    const orderedScore = sideScore + (4 - centerDistance)
    const sideRate = probability(sideScore)
    candidates.push({
      move,
      row: Math.floor(move / 5),
      col: move % 5,
      layer,
      score: orderedScore,
      sideWinRate: sideRate,
      blackWinRate: side === 'B' ? sideRate : 1 - sideRate,
      tag: wins ? 'win' : blocks ? 'block' : null,
    })
  }

  candidates.sort((left, right) => right.score - left.score)
  const best = candidates[0]
  const elapsedMs = performance.now() - startedAt
  return {
    ok: true,
    engineVersion: 0,
    engineMode: 'preview',
    ply: request.moves.length,
    sideToMove: side,
    heights: board.heights,
    stacks: board.stacks,
    terminal: false,
    winner: null,
    bestMove: best.move,
    score: best.score,
    blackWinRate: best.blackWinRate,
    candidates: candidates.sort((left, right) => left.move - right.move),
    stats: {
      depth: 1,
      elapsedMs,
      nodes: candidates.length,
      nps: elapsedMs > 0 ? (candidates.length * 1000) / elapsedMs : 0,
      ttHitRate: 0,
    },
  }
}
