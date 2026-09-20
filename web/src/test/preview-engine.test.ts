import { describe, expect, it } from 'vitest'
import { previewAnalyze } from '../engine/preview-engine'

const limits = { maxDepth: 9, timeLimitMs: 0, tableMegabytes: 16 }

describe('preview engine contract', () => {
  it('returns every legal opening move', () => {
    const result = previewAnalyze({ moves: [], ...limits })
    expect(result.candidates).toHaveLength(25)
    expect(result.bestMove).toBeGreaterThanOrEqual(0)
    expect(result.bestMove).toBeLessThan(25)
    expect(result.sideToMove).toBe('B')
  })

  it('recognizes a completed horizontal line', () => {
    const result = previewAnalyze({ moves: [0, 5, 1, 6, 2, 7, 3], ...limits })
    expect(result.terminal).toBe(true)
    expect(result.winner).toBe('B')
    expect(result.candidates).toHaveLength(0)
  })

  it('omits full columns from candidate analysis', () => {
    const result = previewAnalyze({
      moves: [0, 0, 0, 0, 0],
      ...limits,
    })
    expect(result.terminal).toBe(false)
    expect(result.heights[0]).toBe(5)
    expect(result.candidates.some((candidate) => candidate.move === 0)).toBe(false)
  })
})
