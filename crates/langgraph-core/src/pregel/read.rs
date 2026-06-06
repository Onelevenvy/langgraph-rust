use std::sync::Arc;
use crate::runnable::Runnable;

/// Specification for a graph node in the Pregel engine.
///
/// This is NOT a Runnable itself — it's a container from which
/// `PregelExecutableTask`s are built during each super-step.
pub struct PregelNode {
    /// Which channels to read as input.
    /// If the node reads a single channel, this is `[channel_name]`.
    /// If it reads multiple, each becomes a key in the input dict.
    pub channels: Vec<String>,
    /// Channels whose updates trigger this node.
    pub triggers: Vec<String>,
    /// The main node logic runnable.
    pub bound: Arc<dyn Runnable>,
    /// Writers executed after `bound` (typically ChannelWrite).
    pub writers: Vec<Arc<dyn Runnable>>,
    /// Optional retry policy name.
    pub retry_policy: Option<String>,
    /// Optional metadata.
    pub tags: Vec<String>,
}

impl PregelNode {
    pub fn new(
        channels: Vec<String>,
        triggers: Vec<String>,
        bound: Arc<dyn Runnable>,
    ) -> Self {
        Self {
            channels,
            triggers,
            bound,
            writers: Vec::new(),
            retry_policy: None,
            tags: Vec::new(),
        }
    }

    /// Add a writer to execute after the node logic.
    pub fn with_writer(mut self, writer: Arc<dyn Runnable>) -> Self {
        self.writers.push(writer);
        self
    }

    /// Get all writers flattened (deduped consecutive ChannelWrites).
    pub fn flat_writers(&self) -> &Vec<Arc<dyn Runnable>> {
        &self.writers
    }
}
