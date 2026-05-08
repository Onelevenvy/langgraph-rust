export type TraceStatus = 'running' | 'success' | 'error' | 'interrupted';
export type SpanType = 'graph_node' | 'llm_generation' | 'tool_call';
export type SpanStatus = 'running' | 'success' | 'error';

export interface TraceSummary {
  id: string;
  name: string;
  status: TraceStatus;
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  span_count: number;
}

export interface SpanMetadata {
  model: string | null;
  provider: string | null;
  tokens_in: number | null;
  tokens_out: number | null;
  total_tokens: number | null;
  tool_name: string | null;
  extra: Record<string, unknown>;
}

export interface Span {
  id: string;
  trace_id: string;
  parent_span_id: string | null;
  name: string;
  span_type: SpanType;
  input: unknown;
  output: unknown | null;
  status: SpanStatus;
  start_time: string;
  end_time: string | null;
  metadata: SpanMetadata;
}

export interface Trace {
  id: string;
  name: string;
  input: unknown;
  output: unknown | null;
  status: TraceStatus;
  start_time: string;
  end_time: string | null;
  metadata: Record<string, unknown>;
}

export interface TraceDetail {
  trace: Trace;
  spans: Span[];
}

export type TracingEvent =
  | { type: 'trace_created'; trace: TraceSummary }
  | { type: 'trace_updated'; trace: TraceSummary }
  | { type: 'span_created'; span: Span }
  | { type: 'span_updated'; span: Span };

const BASE = '';

export async function fetchTraces(params?: {
  status?: string;
  name?: string;
  limit?: number;
  offset?: number;
}): Promise<TraceSummary[]> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set('status', params.status);
  if (params?.name) qs.set('name', params.name);
  if (params?.limit) qs.set('limit', String(params.limit));
  if (params?.offset) qs.set('offset', String(params.offset));
  const url = `${BASE}/api/traces${qs.toString() ? '?' + qs : ''}`;
  const res = await fetch(url);
  return res.json();
}

export async function fetchTraceDetail(traceId: string): Promise<TraceDetail> {
  const res = await fetch(`${BASE}/api/traces/${traceId}`);
  if (!res.ok) throw new Error(`Trace not found: ${traceId}`);
  return res.json();
}

export async function clearTraces(): Promise<void> {
  await fetch(`${BASE}/api/traces`, { method: 'DELETE' });
}

export function connectWebSocket(onEvent: (event: TracingEvent) => void): WebSocket {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws`);
  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as TracingEvent;
      onEvent(event);
    } catch (e) {
      console.error('Failed to parse WS message:', e);
    }
  };
  return ws;
}
