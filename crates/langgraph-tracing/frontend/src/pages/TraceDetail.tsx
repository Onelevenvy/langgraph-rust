import { useState, useEffect } from 'react'
import { fetchTraceDetail } from '../api'
import type { TraceDetail as TraceDetailType, Span, SpanType } from '../api'

const SPAN_TYPE_LABELS: Record<SpanType, string> = {
  graph_node: 'Node',
  llm_generation: 'LLM',
  tool_call: 'Tool',
}

const SPAN_TYPE_COLORS: Record<SpanType, string> = {
  graph_node: '#8b5cf6',
  llm_generation: '#3b82f6',
  tool_call: '#f59e0b',
}

function formatDuration(ms: number | null): string {
  if (ms === null) return '-'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function formatJson(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  return JSON.stringify(value, null, 2)
}

function truncate(s: string, max: number): string {
  if (s.length <= max) return s
  return s.slice(0, max) + '...'
}

interface Props {
  traceId: string
  onBack: () => void
}

export default function TraceDetail({ traceId, onBack }: Props) {
  const [detail, setDetail] = useState<TraceDetailType | null>(null)
  const [loading, setLoading] = useState(true)
  const [expandedSpans, setExpandedSpans] = useState<Set<string>>(new Set())

  useEffect(() => {
    setLoading(true)
    fetchTraceDetail(traceId)
      .then(setDetail)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [traceId])

  const toggleSpan = (spanId: string) => {
    setExpandedSpans(prev => {
      const next = new Set(prev)
      if (next.has(spanId)) next.delete(spanId)
      else next.add(spanId)
      return next
    })
  }

  if (loading) return <div className="loading">Loading trace...</div>
  if (!detail) return <div className="empty">Trace not found</div>

  const { trace, spans } = detail

  const rootSpans = spans.filter(s => !s.parent_span_id)
  const childrenOf = (parentId: string) =>
    spans.filter(s => s.parent_span_id === parentId)

  const traceStart = new Date(trace.start_time).getTime()
  const traceEnd = trace.end_time ? new Date(trace.end_time).getTime() : Date.now()
  const totalMs = traceEnd - traceStart

  function renderSpan(span: Span, depth: number): React.ReactNode {
    const expanded = expandedSpans.has(span.id)
    const children = childrenOf(span.id)
    const hasChildren = children.length > 0
    const spanStart = new Date(span.start_time).getTime()
    const spanEnd = span.end_time ? new Date(span.end_time).getTime() : Date.now()
    const leftPct = ((spanStart - traceStart) / totalMs) * 100
    const widthPct = Math.max(((spanEnd - spanStart) / totalMs) * 100, 0.5)

    return (
      <div key={span.id} className="span-group">
        <div
          className={`span-row ${span.status === 'error' ? 'span-error' : ''}`}
          style={{ paddingLeft: `${depth * 24}px` }}
          onClick={() => toggleSpan(span.id)}
        >
          <div className="span-info">
            <span className={`span-expand ${hasChildren ? 'has-children' : ''}`}>
              {hasChildren ? (expanded ? '▼' : '▶') : '•'}
            </span>
            <span
              className="span-type-badge"
              style={{ backgroundColor: SPAN_TYPE_COLORS[span.span_type] }}
            >
              {SPAN_TYPE_LABELS[span.span_type]}
            </span>
            <span className="span-name">{span.name}</span>
            {span.metadata.tokens_in !== null && (
              <span className="span-tokens">
                {span.metadata.tokens_in} → {span.metadata.tokens_out} tokens
              </span>
            )}
            <span className="span-duration">
              {formatDuration(
                span.end_time
                  ? new Date(span.end_time).getTime() - new Date(span.start_time).getTime()
                  : null
              )}
            </span>
          </div>
          <div className="span-waterfall-bar">
            <div
              className="span-bar"
              style={{
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                backgroundColor: SPAN_TYPE_COLORS[span.span_type],
              }}
            />
          </div>
        </div>
        {expanded && (
          <div className="span-detail" style={{ paddingLeft: depth * 24 + 24 }}>
            {span.metadata.model && (
              <div className="detail-field">
                <span className="field-label">Model:</span> {span.metadata.model}
              </div>
            )}
            {span.metadata.tool_name && (
              <div className="detail-field">
                <span className="field-label">Tool:</span> {span.metadata.tool_name}
              </div>
            )}
            {span.metadata.total_tokens !== null && (
              <div className="detail-field">
                <span className="field-label">Total Tokens:</span> {span.metadata.total_tokens}
              </div>
            )}
            <div className="detail-section">
              <div className="field-label">Input:</div>
              <pre className="json-block">{truncate(formatJson(span.input), 2000)}</pre>
            </div>
            {span.output !== null && (
              <div className="detail-section">
                <div className="field-label">Output:</div>
                <pre className="json-block">{truncate(formatJson(span.output), 2000)}</pre>
              </div>
            )}
          </div>
        )}
        {expanded && hasChildren && (
          <div className="span-children">
            {children.map(child => renderSpan(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="trace-detail">
      <div className="detail-header">
        <button className="btn btn-back" onClick={onBack}>← Back</button>
        <h2>{trace.name}</h2>
        <span className={`status-badge status-${trace.status}`}>{trace.status}</span>
        <span className="trace-time">
          {formatDuration(
            trace.end_time
              ? new Date(trace.end_time).getTime() - new Date(trace.start_time).getTime()
              : null
          )}
        </span>
      </div>

      <div className="detail-meta">
        <div className="meta-item">
          <span className="field-label">Trace ID:</span>
          <span className="mono">{trace.id}</span>
        </div>
        <div className="meta-item">
          <span className="field-label">Spans:</span> {spans.length}
        </div>
        <div className="meta-item">
          <span className="field-label">Started:</span>
          {new Date(trace.start_time).toLocaleString()}
        </div>
      </div>

      {trace.input !== null && trace.input !== undefined && (
        <div className="detail-section">
          <div className="field-label">Input:</div>
          <pre className="json-block">{truncate(formatJson(trace.input), 3000)}</pre>
        </div>
      )}
      {trace.output !== null && trace.output !== undefined && (
        <div className="detail-section">
          <div className="field-label">Output:</div>
          <pre className="json-block">{truncate(formatJson(trace.output), 3000)}</pre>
        </div>
      )}

      <h3>Spans ({spans.length})</h3>
      <div className="span-tree">
        {rootSpans.map(span => renderSpan(span, 0))}
      </div>
    </div>
  )
}
