import { useState, useEffect, useCallback } from 'react'
import { fetchTraces, clearTraces, connectWebSocket } from './api'
import type { TraceSummary, TracingEvent } from './api'
import TracesList from './pages/TracesList'
import TraceDetail from './pages/TraceDetail'

function App() {
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [nameFilter, setNameFilter] = useState('')
  const [loading, setLoading] = useState(true)

  const loadTraces = useCallback(async () => {
    try {
      const data = await fetchTraces({
        status: statusFilter || undefined,
        name: nameFilter || undefined,
      })
      setTraces(data)
    } catch (e) {
      console.error('Failed to load traces:', e)
    } finally {
      setLoading(false)
    }
  }, [statusFilter, nameFilter])

  useEffect(() => {
    loadTraces()
  }, [loadTraces])

  useEffect(() => {
    const ws = connectWebSocket((event: TracingEvent) => {
      if (event.type === 'trace_created') {
        setTraces(prev => [event.trace, ...prev])
      } else if (event.type === 'trace_updated') {
        setTraces(prev => prev.map(t => t.id === event.trace.id ? event.trace : t))
      }
    })
    return () => ws.close()
  }, [])

  const handleClear = async () => {
    await clearTraces()
    setTraces([])
    setSelectedTraceId(null)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1 onClick={() => setSelectedTraceId(null)} style={{ cursor: 'pointer' }}>
            LangGraph Tracing
          </h1>
        </div>
        <div className="header-right">
          <span className="trace-count">{traces.length} traces</span>
          <button className="btn btn-danger" onClick={handleClear}>Clear All</button>
        </div>
      </header>
      <main className="app-main">
        {selectedTraceId ? (
          <TraceDetail
            traceId={selectedTraceId}
            onBack={() => setSelectedTraceId(null)}
          />
        ) : (
          <TracesList
            traces={traces}
            loading={loading}
            statusFilter={statusFilter}
            nameFilter={nameFilter}
            onStatusFilterChange={setStatusFilter}
            onNameFilterChange={setNameFilter}
            onSelectTrace={setSelectedTraceId}
            onRefresh={loadTraces}
          />
        )}
      </main>
    </div>
  )
}

export default App
