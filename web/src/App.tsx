import { useCallback, useEffect, useMemo, useRef } from 'react'
import { BoardScene } from './scene/BoardScene'
import { EngineStats } from './components/EngineStats'
import { MoveTimeline } from './components/MoveTimeline'
import { WinRateChart } from './components/WinRateChart'
import { engineClient } from './engine/engine-client'
import { useGameStore } from './store/game-store'
import {
  DIFFICULTIES,
  moveLabel,
  sideLabel,
  type AnalysisResult,
  type Side,
} from './types'

function wait(duration: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, duration))
}

export default function App() {
  const state = useGameStore()
  const operation = useRef(0)

  const analyze = useCallback(async (moves: number[], token: number) => {
    const { difficulty } = useGameStore.getState()
    const result = await engineClient.analyze({
      moves,
      maxDepth: difficulty.maxDepth,
      timeLimitMs: difficulty.timeLimitMs,
      tableMegabytes: 32,
    })
    if (token !== operation.current) throw new Error('cancelled')
    return result
  }, [])

  const finishPosition = useCallback((analysis: AnalysisResult) => {
    useGameStore.getState().setAnalysis(analysis)
  }, [])

  const playAiMove = useCallback(async (analysis: AnalysisResult, token: number) => {
    if (analysis.terminal || analysis.bestMove < 0) {
      finishPosition(analysis)
      return
    }
    finishPosition(analysis)
    await wait(260)
    if (token !== operation.current) return
    const nextMoves = useGameStore.getState().recordMove(analysis.bestMove, analysis)
    const nextAnalysis = await analyze(nextMoves, token)
    finishPosition(nextAnalysis)
  }, [analyze, finishPosition])

  const continueFrom = useCallback(async (moves: number[], token: number) => {
    const analysis = await analyze(moves, token)
    const { humanSide } = useGameStore.getState()
    if (!analysis.terminal && analysis.sideToMove !== humanSide) {
      await playAiMove(analysis, token)
    } else {
      finishPosition(analysis)
    }
  }, [analyze, finishPosition, playAiMove])

  const startGame = useCallback((humanSide: Side) => {
    const token = ++operation.current
    engineClient.restart()
    useGameStore.getState().begin(humanSide)
    void continueFrom([], token).catch((error: unknown) => {
      if (token !== operation.current) return
      useGameStore.getState().setError(
        error instanceof Error ? error.message : '无法启动引擎',
      )
    })
  }, [continueFrom])

  useEffect(() => {
    startGame('W')
    return () => {
      operation.current += 1
      engineClient.restart()
    }
  }, [startGame])

  const handleHumanMove = useCallback((move: number) => {
    const current = useGameStore.getState()
    if (
      current.status !== 'human' ||
      current.reviewPly !== null ||
      !current.currentAnalysis ||
      current.currentAnalysis.sideToMove !== current.humanSide ||
      !current.currentAnalysis.candidates.some((candidate) => candidate.move === move)
    ) return

    const token = ++operation.current
    const nextMoves = current.recordMove(move, current.currentAnalysis)
    void continueFrom(nextMoves, token).catch((error: unknown) => {
      if (token !== operation.current) return
      useGameStore.getState().setError(
        error instanceof Error ? error.message : '搜索失败',
      )
    })
  }, [continueFrom])

  const handleUndo = useCallback(() => {
    const current = useGameStore.getState()
    let humanIndex = -1
    for (let index = current.history.length - 1; index >= 0; index -= 1) {
      if (current.history[index].side === current.humanSide) {
        humanIndex = index
        break
      }
    }
    if (humanIndex < 0) return

    const token = ++operation.current
    engineClient.restart()
    const entry = current.history[humanIndex]
    const moves = [...entry.movesBefore]
    const history = current.history.slice(0, humanIndex)
    current.replacePosition(moves, history)
    void continueFrom(moves, token).catch((error: unknown) => {
      if (token !== operation.current) return
      useGameStore.getState().setError(
        error instanceof Error ? error.message : '悔棋后重新分析失败',
      )
    })
  }, [continueFrom])

  const handleDifficultyChange = useCallback((difficultyId: string) => {
    const difficulty = DIFFICULTIES.find((item) => item.id === difficultyId)
    if (!difficulty) return

    const current = useGameStore.getState()
    current.setDifficulty(difficulty)
    const token = ++operation.current
    engineClient.restart()
    current.setStatus('loading')
    void continueFrom([...current.moves], token).catch((error: unknown) => {
      if (token !== operation.current) return
      useGameStore.getState().setError(
        error instanceof Error ? error.message : '切换棋力后重新分析失败',
      )
    })
  }, [continueFrom])

  const reviewEntry = useMemo(
    () => state.reviewPly === null
      ? null
      : state.history.find((entry) => entry.ply === state.reviewPly) ?? null,
    [state.history, state.reviewPly],
  )
  const displayMoves = reviewEntry?.movesBefore ?? state.moves
  const displayAnalysis = reviewEntry?.analysis ?? state.currentAnalysis
  const selectedMove = reviewEntry?.move ?? null
  const blackRate = reviewEntry?.blackWinRate
    ?? state.currentAnalysis?.blackWinRate
    ?? state.initialBlackWinRate
  const sideRate = displayAnalysis?.sideToMove === 'W' ? 1 - blackRate : blackRate
  const interactive = state.status === 'human' && state.reviewPly === null

  const statusText = (() => {
    if (state.reviewPly !== null) return `复盘第 ${state.reviewPly} 手 · 显示落子前候选`
    if (state.status === 'loading') return '正在装载棋局…'
    if (state.status === 'thinking') return '对手正在推演变化'
    if (state.status === 'human') return `${sideLabel(state.humanSide)}行棋`
    if (state.status === 'terminal') {
      return state.winner ? `${sideLabel(state.winner)}胜` : '和棋'
    }
    return state.error ?? '引擎暂不可用'
  })()

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-seal">重</span>
          <div>
            <p>GRAVITY FOUR</p>
            <h1>棋院复盘室</h1>
          </div>
        </div>
        <div className="topbar-status">
          <span className={`status-dot ${state.status === 'thinking' || state.status === 'loading' ? 'is-thinking' : ''}`} />
          <strong>{statusText}</strong>
          <small>第 {state.moves.length + 1} 手</small>
        </div>
        <div className="topbar-actions">
          <button type="button" onClick={handleUndo} disabled={!state.history.some((entry) => entry.side === state.humanSide)}>
            悔棋一回合
          </button>
          <button type="button" className="primary-action" onClick={() => startGame(state.humanSide)}>
            新对局
          </button>
        </div>
      </header>

      <main className="game-layout">
        <section className="board-panel">
          <div className="board-toolbar">
            <div className="segmented-control" aria-label="棋盘视角">
              {(['play', 'analysis', 'layers'] as const).map((mode) => (
                <button
                  type="button"
                  key={mode}
                  className={state.viewMode === mode ? 'is-active' : ''}
                  onClick={() => state.setViewMode(mode)}
                >
                  {mode === 'play' ? '对局' : mode === 'analysis' ? '分析' : '分层'}
                </button>
              ))}
            </div>
            <label className="heatmap-toggle">
              <input
                type="checkbox"
                checked={state.heatmapEnabled}
                onChange={(event) => state.setHeatmapEnabled(event.target.checked)}
              />
              <span />
              落点胜率
            </label>
          </div>

          <BoardScene
            moves={displayMoves}
            analysis={displayAnalysis}
            heatmapEnabled={state.heatmapEnabled}
            viewMode={state.viewMode}
            interactive={interactive}
            selectedMove={selectedMove}
            animateLastMove={state.reviewPly === null}
            onColumnClick={handleHumanMove}
          />

          {(state.status === 'thinking' || state.status === 'loading') && state.reviewPly === null && (
            <div className="thinking-ribbon">
              <i /><span>Minimax 正在搜索完整根节点</span><b>请稍候</b>
            </div>
          )}
        </section>

        <aside className="analysis-panel">
          <div className="score-header">
            <div className="score-side is-black">
              <span>黑方</span>
              <strong>{Math.round(blackRate * 100)}<small>%</small></strong>
            </div>
            <div className="score-balance" aria-hidden="true">
              <span style={{ width: `${blackRate * 100}%` }} />
            </div>
            <div className="score-side is-white">
              <span>白方</span>
              <strong>{Math.round((1 - blackRate) * 100)}<small>%</small></strong>
            </div>
          </div>

          <section className="chart-card">
            <div className="panel-heading">
              <div><span>形势</span><h2>胜率走势</h2></div>
              <small>点击节点复盘</small>
            </div>
            <WinRateChart
              history={state.history}
              initialBlackWinRate={state.initialBlackWinRate}
              selectedPly={state.reviewPly}
              onSelectPly={state.setReviewPly}
            />
          </section>

          <section className="position-card">
            <div className="panel-heading">
              <div>
                <span>{reviewEntry ? `第 ${reviewEntry.ply} 手` : '当前局面'}</span>
                <h2>{reviewEntry ? `${sideLabel(reviewEntry.side)} · ${moveLabel(reviewEntry.move)}` : statusText}</h2>
              </div>
              <span className={`engine-badge ${displayAnalysis?.engineMode === 'preview' ? 'is-preview' : ''}`}>
                {displayAnalysis?.engineMode === 'wasm'
                  ? 'WASM'
                  : displayAnalysis?.engineMode === 'preview'
                    ? '预览引擎'
                    : '装载中'}
              </span>
            </div>
            <div className="position-evaluation">
              <span>{displayAnalysis ? `${sideLabel(displayAnalysis.sideToMove)}估算` : '等待分析'}</span>
              <strong>{Math.round(sideRate * 100)}%</strong>
              <small>引擎估算胜率，不代表统计概率</small>
            </div>
            <EngineStats analysis={displayAnalysis} />
          </section>

          <section className="settings-card">
            <div className="setting-row">
              <span>我执</span>
              <div className="side-picker">
                {(['B', 'W'] as const).map((side) => (
                  <button
                    type="button"
                    key={side}
                    className={state.humanSide === side ? 'is-active' : ''}
                    onClick={() => startGame(side)}
                  >
                    <i className={side === 'B' ? 'black-stone' : 'white-stone'} />
                    {side === 'B' ? '黑' : '白'}
                  </button>
                ))}
              </div>
            </div>
            <div className="setting-row">
              <span>棋力</span>
              <select
                value={state.difficulty.id}
                onChange={(event) => handleDifficultyChange(event.target.value)}
              >
                {DIFFICULTIES.map((difficulty) => (
                  <option key={difficulty.id} value={difficulty.id}>
                    {difficulty.label} · {difficulty.caption}
                  </option>
                ))}
              </select>
            </div>
          </section>
        </aside>
      </main>

      <MoveTimeline
        history={state.history}
        selectedPly={state.reviewPly}
        onSelectPly={state.setReviewPly}
        onReturnLive={() => state.setReviewPly(null)}
      />

      {state.currentAnalysis?.engineMode === 'preview' && (
        <div className="preview-notice">
          WASM 尚未装载，当前使用轻量预览引擎；界面与规则可正常体验，完整 Minimax 构建后自动切换。
        </div>
      )}
    </div>
  )
}
