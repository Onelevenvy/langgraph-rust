import type { TraceSummary, TraceStatus } from '../api'

const STATUS_COLORS: Record<TraceStatus, string> = {
  running: '#3b82f6',
  success: '#22c55e',
  error: '#ef4444',
  interrupted: '#f59e0b',
}

function formatDuration(ms: number | null): string {
  if (ms === null) return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function formatTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleTimeString() + '.' + String(d.getMilliseconds()).padStart(3, '0')
}

interface Props {
  traces: TraceSummary[]
  loading: boolean
  statusFilter: string
  nameFilter: string
  onStatusFilterChange: (v: string) => void
  onNameFilterChange: (v: string) => void
  onSelectTrace: (id: string) => void
  onRefresh: () => void
}

export default function TracesList({
  traces, loading, statusFilter, nameFilter,
  onStatusFilterChange, onNameFilterChange, onSelectTrace, onRefresh,
}: Props) {
  return (
    <div className="traces-list">
      <div className="filters">
        <input
          type="text"
          placeholder="Search by name..."
          value={nameFilter}
          onChange={e => onNameFilterChange(e.target.value)}
          className="filter-input"
        />
        <select
          value={statusFilter}
          onChange={e => onStatusFilterChange(e.target.value)}
          className="filter-select"
        >
          <option value="">All Status</option>
          <option value="running">Running</option>
          <option value="success">Success</option>
          <option value="error">Error</option>
          <option value="interrupted">Interrupted</option>
        </select>
        <button className="btn" onClick={onRefresh}>Refresh</button>
      </div>

      {loading ? (
        <div className="loading">Loading traces...</div>
      ) : traces.length === 0 ? (
        <div className="empty">
          <p>No traces yet.</p>
          <p className="empty-hint">Run a graph with tracing enabled to see results here.</p>
        </div>
      ) : (
        <table className="traces-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Name</th>
              <th>Duration</th>
              <th>Spans</th>
              <th>Start Time</th>
            </tr>
          </thead>
          <tbody>
            {traces.map(trace => (
              <tr
                key={trace.id}
                className="trace-row"
                onClick={() => onSelectTrace(trace.id)}
              >
                <td>
                  <span
                    className="status-badge"
                    style={{ backgroundColor: STATUS_COLORS[trace.status] }}
                  >
                    {trace.status}
                  </span>
                </td>
                <td className="trace-name">{trace.name}</td>
                <td>{formatDuration(trace.duration_ms)}</td>
                <td>{trace.span_count}</td>
                <td>{formatTime(trace.start_time)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
