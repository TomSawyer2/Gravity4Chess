import type { HistoryEntry } from '../types'
import { moveLabel } from '../types'

interface MoveTimelineProps {
  history: HistoryEntry[]
  selectedPly: number | null
  onSelectPly: (ply: number) => void
  onReturnLive: () => void
}

export function MoveTimeline({
  history,
  selectedPly,
  onSelectPly,
  onReturnLive,
}: MoveTimelineProps) {
  return (
    <section className="timeline-shell" aria-label="对局手数">
      <div className="timeline-label">
        <span>手数</span>
        <strong>{history.length.toString().padStart(2, '0')}</strong>
      </div>
      <div className="timeline-list">
        {history.length === 0 && <span className="timeline-empty">落子记录会出现在这里</span>}
        {history.map((entry) => (
          <button
            key={entry.ply}
            type="button"
            className={`timeline-move ${selectedPly === entry.ply ? 'is-selected' : ''}`}
            onClick={() => onSelectPly(entry.ply)}
            title={`查看第 ${entry.ply} 手落子前的全部候选胜率`}
          >
            <span className={`timeline-stone ${entry.side === 'B' ? 'is-black' : 'is-white'}`} />
            <small>{entry.ply}</small>
            <b>{moveLabel(entry.move)}</b>
          </button>
        ))}
      </div>
      {selectedPly !== null && (
        <button type="button" className="return-live" onClick={onReturnLive}>
          返回实战
        </button>
      )}
    </section>
  )
}
