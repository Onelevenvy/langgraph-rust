use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Status of a trace (graph run)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Running,
    Success,
    Error,
    Interrupted,
}

/// Type of span
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanType {
    /// Graph node execution
    GraphNode,
    /// LLM API call
    LlmGeneration,
    /// Tool invocation
    ToolCall,
}

/// Status of a span
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    Running,
    Success,
    Error,
}

/// A trace represents a single graph execution run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub id: String,
    pub name: String,
    pub input: JsonValue,
    pub output: Option<JsonValue>,
    pub status: TraceStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, JsonValue>,
}

impl Trace {
    pub fn new(id: String, name: String, input: JsonValue) -> Self {
        Self {
            id,
            name,
            input,
            output: None,
            status: TraceStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            metadata: HashMap::new(),
        }
    }

    pub fn duration_ms(&self) -> Option<u64> {
        let end = self.end_time.unwrap_or_else(Utc::now);
        let dur = (end - self.start_time).num_milliseconds();
        if dur < 0 { Some(0) } else { Some(dur as u64) }
    }

    pub fn finish(&mut self, output: JsonValue, status: TraceStatus) {
        self.output = Some(output);
        self.status = status;
        self.end_time = Some(Utc::now());
    }
}

/// A span represents a single unit of work within a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    pub id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub span_type: SpanType,
    pub input: JsonValue,
    pub output: Option<JsonValue>,
    pub status: SpanStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    /// Type-specific metadata
    pub metadata: SpanMetadata,
}

impl Span {
    pub fn new(
        id: String,
        trace_id: String,
        parent_span_id: Option<String>,
        name: String,
        span_type: SpanType,
        input: JsonValue,
    ) -> Self {
        Self {
            id,
            trace_id,
            parent_span_id,
            name,
            span_type,
            input,
            output: None,
            status: SpanStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            metadata: SpanMetadata::default(),
        }
    }

    pub fn duration_ms(&self) -> Option<u64> {
        let end = self.end_time.unwrap_or_else(Utc::now);
        let dur = (end - self.start_time).num_milliseconds();
        if dur < 0 { Some(0) } else { Some(dur as u64) }
    }

    pub fn finish(&mut self, output: JsonValue, status: SpanStatus) {
        self.output = Some(output);
        self.status = status;
        self.end_time = Some(Utc::now());
    }
}

/// Type-specific metadata for spans
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpanMetadata {
    /// Model name for LLM generations (e.g., "gpt-4o")
    pub model: Option<String>,
    /// Provider name (e.g., "openai")
    pub provider: Option<String>,
    /// Input token count
    pub tokens_in: Option<u32>,
    /// Output token count
    pub tokens_out: Option<u32>,
    /// Total token count
    pub total_tokens: Option<u32>,
    /// Tool name for tool call spans
    pub tool_name: Option<String>,
    /// Custom key-value metadata
    pub extra: HashMap<String, JsonValue>,
}

/// Summary of a trace for the list view (without full input/output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub id: String,
    pub name: String,
    pub status: TraceStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub span_count: usize,
}

impl From<&Trace> for TraceSummary {
    fn from(trace: &Trace) -> Self {
        Self {
            id: trace.id.clone(),
            name: trace.name.clone(),
            status: trace.status,
            start_time: trace.start_time,
            end_time: trace.end_time,
            duration_ms: trace.duration_ms(),
            span_count: 0,
        }
    }
}

/// Trace detail with all spans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDetail {
    pub trace: Trace,
    pub spans: Vec<Span>,
}
