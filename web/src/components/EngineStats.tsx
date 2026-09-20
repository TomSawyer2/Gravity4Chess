import type { AnalysisResult } from '../types'

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}m`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return `${Math.round(value)}`
}

export function EngineStats({ analysis }: { analysis: AnalysisResult | null }) {
  const stats = analysis?.stats
  return (
    <div className="engine-stats">
      <div><span>深度</span><strong>{stats?.depth ?? '—'}</strong></div>
      <div><span>单步</span><strong>{stats ? `${stats.elapsedMs.toFixed(stats.elapsedMs < 10 ? 1 : 0)} ms` : '—'}</strong></div>
      <div><span>节点</span><strong>{stats ? formatCount(stats.nodes) : '—'}</strong></div>
      <div><span>NPS</span><strong>{stats ? formatCount(stats.nps) : '—'}</strong></div>
    </div>
  )
}
