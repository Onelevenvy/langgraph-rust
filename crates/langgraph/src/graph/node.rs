use std::sync::Arc;
use serde_json::Value as JsonValue;
use crate::runnable::Runnable;
use crate::types::RetryPolicy;

/// Specification for a graph node.
///
/// Holds the node's runnable, metadata, and execution options.
#[derive(Clone)]
pub struct StateNodeSpec {
    /// Name of this node.
    pub name: String,
    /// The executable logic for this node.
    pub runnable: Arc<dyn Runnable>,
    /// Optional retry policy.
    pub retry_policy: Option<RetryPolicy>,
    /// Optional metadata.
    pub metadata: Option<JsonValue>,
    /// Static destinations inferred from Command return types.
    /// If set, the node can only route to these targets.
    pub ends: Option<Vec<String>>,
}

impl StateNodeSpec {
    pub fn new(name: impl Into<String>, runnable: Arc<dyn Runnable>) -> Self {
        Self {
            name: name.into(),
            runnable,
            retry_policy: None,
            metadata: None,
            ends: None,
        }
    }
}
