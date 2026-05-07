use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use crate::types::GraphInterrupt;

/// Core Runnable trait — the universal execution abstraction.
///
/// All input/output is type-erased to `serde_json::Value` to match
/// the channel layer's type erasure strategy.
#[async_trait]
pub trait Runnable: Send + Sync {
    /// Execute synchronously.
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError>;

    /// Execute asynchronously.
    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError>;

    /// Human-readable name for tracing/debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

/// Errors from Runnable execution.
#[derive(Debug, thiserror::Error)]
pub enum RunnableError {
    #[error("node error: {0}")]
    Node(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("channel error: {0}")]
    Channel(#[from] langgraph_checkpoint::error::ChannelError),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("runner error: {0}")]
    Runner(String),

    #[error("graph interrupt")]
    Interrupt(GraphInterrupt),
}
