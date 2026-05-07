use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("recursion limit exceeded")]
    RecursionLimit,

    #[error("invalid update for channel '{channel}': {message}")]
    InvalidUpdate { channel: String, message: String },

    #[error("graph interrupted")]
    GraphInterrupt,

    #[error("empty channel: {0}")]
    EmptyChannel(String),

    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("invalid graph: {0}")]
    InvalidGraph(String),

    #[error("checkpoint error: {0}")]
    Checkpoint(#[from] langgraph_checkpoint::error::CheckpointError),

    #[error("channel error: {0}")]
    Channel(#[from] langgraph_checkpoint::error::ChannelError),

    #[error("serialization error: {0}")]
    Serde(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
