import { afterEach, describe, expect, it, vi } from 'vitest'
import { EngineClient } from '../engine/engine-client'

class FailingWorker {
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  onmessageerror: (() => void) | null = null

  postMessage(): void {
    queueMicrotask(() => {
      this.onerror?.({ preventDefault: vi.fn() } as unknown as ErrorEvent)
    })
  }

  terminate(): void {}
}

describe('engine client resilience', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('falls back to the preview engine when the worker cannot start', async () => {
    vi.stubGlobal('Worker', FailingWorker)
    const client = new EngineClient()
    const result = await client.analyze({
      moves: [],
      maxDepth: 9,
      timeLimitMs: 400,
      tableMegabytes: 16,
    })

    expect(result.engineMode).toBe('preview')
    expect(result.candidates).toHaveLength(25)
    expect(result.bestMove).toBeGreaterThanOrEqual(0)
  })
})
