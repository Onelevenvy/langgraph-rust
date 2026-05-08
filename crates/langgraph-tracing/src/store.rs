use crate::types::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trait for trace storage backends
pub trait TracingStore: Send + Sync {
    fn create_trace(&self, trace: Trace);
    fn update_trace(&self, trace: Trace);
    fn get_trace(&self, trace_id: &str) -> Option<TraceDetail>;
    fn list_traces(&self, filter: &TraceFilter) -> Vec<TraceSummary>;
    fn add_span(&self, span: Span);
    fn update_span(&self, span: Span);
    fn clear(&self);
}

/// Filter for listing traces
#[derive(Debug, Clone, Default)]
pub struct TraceFilter {
    pub status: Option<TraceStatus>,
    pub name_contains: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// In-memory implementation of TracingStore
#[derive(Clone)]
pub struct InMemoryTracingStore {
    traces: Arc<RwLock<HashMap<String, Trace>>>,
    spans: Arc<RwLock<HashMap<String, Vec<Span>>>>,
}

impl InMemoryTracingStore {
    pub fn new() -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
            spans: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryTracingStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TracingStore for InMemoryTracingStore {
    fn create_trace(&self, trace: Trace) {
        let mut traces = self.traces.write().unwrap();
        traces.insert(trace.id.clone(), trace);
    }

    fn update_trace(&self, trace: Trace) {
        let mut traces = self.traces.write().unwrap();
        traces.insert(trace.id.clone(), trace);
    }

    fn get_trace(&self, trace_id: &str) -> Option<TraceDetail> {
        let traces = self.traces.read().unwrap();
        let spans = self.spans.read().unwrap();
        traces.get(trace_id).map(|trace| TraceDetail {
            trace: trace.clone(),
            spans: spans.get(trace_id).cloned().unwrap_or_default(),
        })
    }

    fn list_traces(&self, filter: &TraceFilter) -> Vec<TraceSummary> {
        let traces = self.traces.read().unwrap();
        let spans = self.spans.read().unwrap();

        let mut summaries: Vec<TraceSummary> = traces
            .values()
            .filter(|t| {
                if let Some(ref status) = filter.status {
                    if &t.status != status {
                        return false;
                    }
                }
                if let Some(ref name) = filter.name_contains {
                    if !t.name.contains(name.as_str()) {
                        return false;
                    }
                }
                true
            })
            .map(|t| {
                let mut summary = TraceSummary::from(t);
                summary.span_count = spans.get(&t.id).map(|s| s.len()).unwrap_or(0);
                summary
            })
            .collect();

        // Sort newest first
        summaries.sort_by(|a, b| b.start_time.cmp(&a.start_time));

        let offset = filter.offset.unwrap_or(0);
        let limit = filter.limit.unwrap_or(summaries.len());

        summaries
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect()
    }

    fn add_span(&self, span: Span) {
        let mut spans = self.spans.write().unwrap();
        spans
            .entry(span.trace_id.clone())
            .or_default()
            .push(span);
    }

    fn update_span(&self, span: Span) {
        let mut spans = self.spans.write().unwrap();
        if let Some(trace_spans) = spans.get_mut(&span.trace_id) {
            if let Some(existing) = trace_spans.iter_mut().find(|s| s.id == span.id) {
                *existing = span;
            }
        }
    }

    fn clear(&self) {
        self.traces.write().unwrap().clear();
        self.spans.write().unwrap().clear();
    }
}
