use crate::event_bus::{EventBus, TracingEvent};
use crate::store::TracingStore;
use crate::types::*;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use uuid::Uuid;

/// Observer that records graph execution traces and spans.
///
/// Can be used manually or integrated with the streaming system
/// to automatically capture graph runs and node executions.
pub struct TracingGraphObserver {
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
    /// The current active trace (if any)
    current_trace_id: Option<String>,
}

impl TracingGraphObserver {
    pub fn new(store: Arc<dyn TracingStore>, event_bus: EventBus) -> Self {
        Self {
            store,
            event_bus,
            current_trace_id: None,
        }
    }

    /// Start observing a new graph run
    pub fn on_graph_start(&mut self, name: &str, input: JsonValue) -> String {
        let trace_id = Uuid::new_v4().to_string();
        let trace = Trace::new(trace_id.clone(), name.to_string(), input);

        let summary = TraceSummary::from(&trace);
        self.store.create_trace(trace);
        self.event_bus.publish(TracingEvent::TraceCreated { trace: summary });

        self.current_trace_id = Some(trace_id.clone());
        trace_id
    }

    /// Record graph run completion
    pub fn on_graph_end(&self, trace_id: &str, output: JsonValue, status: TraceStatus) {
        if let Some(mut detail) = self.store.get_trace(trace_id) {
            detail.trace.finish(output, status);
            let summary = TraceSummary::from(&detail.trace);
            self.store.update_trace(detail.trace);
            self.event_bus.publish(TracingEvent::TraceUpdated { trace: summary });
        }
    }

    /// Start a node execution span
    pub fn on_node_start(
        &self,
        trace_id: &str,
        parent_span_id: Option<&str>,
        node_name: &str,
        input: JsonValue,
    ) -> String {
        let span_id = Uuid::new_v4().to_string();
        let span = Span::new(
            span_id.clone(),
            trace_id.to_string(),
            parent_span_id.map(|s| s.to_string()),
            node_name.to_string(),
            SpanType::GraphNode,
            input,
        );

        self.store.add_span(span.clone());
        self.event_bus.publish(TracingEvent::SpanCreated { span });
        span_id
    }

    /// Record node execution completion
    pub fn on_node_end(&self, span_id: &str, trace_id: &str, output: JsonValue, success: bool) {
        if let Some(detail) = self.store.get_trace(trace_id) {
            if let Some(mut span) = detail.spans.into_iter().find(|s| s.id == span_id) {
                let status = if success { SpanStatus::Success } else { SpanStatus::Error };
                span.finish(output, status);
                self.store.update_span(span.clone());
                self.event_bus.publish(TracingEvent::SpanUpdated { span });
            }
        }
    }

    /// Record an LLM generation span
    pub fn on_llm_start(
        &self,
        trace_id: &str,
        parent_span_id: Option<&str>,
        model: &str,
        input: JsonValue,
    ) -> String {
        let span_id = Uuid::new_v4().to_string();
        let mut span = Span::new(
            span_id.clone(),
            trace_id.to_string(),
            parent_span_id.map(|s| s.to_string()),
            model.to_string(),
            SpanType::LlmGeneration,
            input,
        );
        span.metadata.model = Some(model.to_string());

        self.store.add_span(span.clone());
        self.event_bus.publish(TracingEvent::SpanCreated { span });
        span_id
    }

    /// Record LLM generation completion with token usage
    pub fn on_llm_end(
        &self,
        span_id: &str,
        trace_id: &str,
        output: JsonValue,
        success: bool,
        tokens_in: Option<u32>,
        tokens_out: Option<u32>,
    ) {
        if let Some(detail) = self.store.get_trace(trace_id) {
            if let Some(mut span) = detail.spans.into_iter().find(|s| s.id == span_id) {
                let status = if success { SpanStatus::Success } else { SpanStatus::Error };
                span.finish(output, status);
                span.metadata.tokens_in = tokens_in;
                span.metadata.tokens_out = tokens_out;
                span.metadata.total_tokens = match (tokens_in, tokens_out) {
                    (Some(a), Some(b)) => Some(a + b),
                    _ => None,
                };
                self.store.update_span(span.clone());
                self.event_bus.publish(TracingEvent::SpanUpdated { span });
            }
        }
    }

    /// Record a tool call span
    pub fn on_tool_start(
        &self,
        trace_id: &str,
        parent_span_id: Option<&str>,
        tool_name: &str,
        input: JsonValue,
    ) -> String {
        let span_id = Uuid::new_v4().to_string();
        let mut span = Span::new(
            span_id.clone(),
            trace_id.to_string(),
            parent_span_id.map(|s| s.to_string()),
            tool_name.to_string(),
            SpanType::ToolCall,
            input,
        );
        span.metadata.tool_name = Some(tool_name.to_string());

        self.store.add_span(span.clone());
        self.event_bus.publish(TracingEvent::SpanCreated { span });
        span_id
    }

    /// Record tool call completion
    pub fn on_tool_end(&self, span_id: &str, trace_id: &str, output: JsonValue, success: bool) {
        if let Some(detail) = self.store.get_trace(trace_id) {
            if let Some(mut span) = detail.spans.into_iter().find(|s| s.id == span_id) {
                let status = if success { SpanStatus::Success } else { SpanStatus::Error };
                span.finish(output, status);
                self.store.update_span(span.clone());
                self.event_bus.publish(TracingEvent::SpanUpdated { span });
            }
        }
    }

    pub fn store(&self) -> &dyn TracingStore {
        self.store.as_ref()
    }

    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }
}
