use std::collections::HashMap;
use std::sync::Arc;
use crate::runnable::Runnable;

/// Specification for a conditional edge (branch).
///
/// The `path` runnable evaluates the current state and returns one or more
/// routing keys. These keys are mapped to destination node names via `ends`.
#[derive(Clone)]
pub struct BranchSpec {
    /// The runnable that evaluates the condition.
    /// Returns a JSON value that is used as the routing key.
    pub path: Arc<dyn Runnable>,
    /// Maps routing keys to destination node names.
    /// If `None`, the routing key itself is used as the node name.
    pub ends: Option<HashMap<String, String>>,
}

impl BranchSpec {
    pub fn new(path: Arc<dyn Runnable>, ends: Option<HashMap<String, String>>) -> Self {
        Self { path, ends }
    }

    /// Resolve a routing key to a destination node name.
    pub fn resolve(&self, key: &str) -> Option<String> {
        match &self.ends {
            Some(map) => map.get(key).cloned(),
            None => Some(key.to_string()),
        }
    }
}
