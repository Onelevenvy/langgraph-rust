use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use crate::types::StreamMode;

/// A single chunk emitted by the graph's streaming interface (v2 format).
///
/// Discriminated on the `mode` field. The `ns` field carries the namespace
/// path (empty for top-level graphs). The `data` field carries the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPart {
    /// The stream mode that produced this chunk.
    pub mode: StreamMode,
    /// Namespace path (empty for top-level graph).
    pub ns: Vec<String>,
    /// The payload data.
    pub data: JsonValue,
}

impl StreamPart {
    pub fn values(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Values, ns, data }
    }

    pub fn updates(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Updates, ns, data }
    }

    pub fn messages(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Messages, ns, data }
    }

    pub fn custom(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Custom, ns, data }
    }

    pub fn tasks(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Tasks, ns, data }
    }

    pub fn checkpoints(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Checkpoints, ns, data }
    }

    pub fn debug(ns: Vec<String>, data: JsonValue) -> Self {
        Self { mode: StreamMode::Debug, ns, data }
    }
}
