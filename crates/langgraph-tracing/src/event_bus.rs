use crate::types::*;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Events emitted by the tracing system for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TracingEvent {
    TraceCreated { trace: TraceSummary },
    TraceUpdated { trace: TraceSummary },
    SpanCreated { span: Span },
    SpanUpdated { span: Span },
}

/// Event bus for real-time WebSocket push
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<TracingEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(256);
        Self { sender }
    }

    pub fn publish(&self, event: TracingEvent) {
        // Ignore send errors (no receivers)
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TracingEvent> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
