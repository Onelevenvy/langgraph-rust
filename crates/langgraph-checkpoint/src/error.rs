use thiserror::Error;

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("serialization error: {0}")]
    Serde(#[from] SerdeError),
    #[error("checkpoint not found")]
    NotFound,
    #[error("storage error: {0}")]
    Storage(String),
    #[error("config error: {0}")]
    Config(String),
}

#[derive(Error, Debug)]
pub enum SerdeError {
    #[error("msgpack error: {0}")]
    Msgpack(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown serialization tag: {0}")]
    UnknownTag(String),
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("not serializable: {0}")]
    NotSerializable(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("store error: {0}")]
    Storage(String),
    #[error("invalid namespace: {0}")]
    InvalidNamespace(String),
}

#[derive(Error, Debug)]
pub enum ChannelError {
    #[error("channel is empty")]
    EmptyChannel,
    #[error("invalid update: {0}")]
    InvalidUpdate(String),
}
