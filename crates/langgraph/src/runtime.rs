use std::sync::Arc;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use langgraph_checkpoint::store::base::BaseStore;

/// StreamWriter sends custom stream chunks to the output stream.
/// Nodes can use this to emit arbitrary data when `stream_mode` includes "custom".
pub type StreamWriter = mpsc::Sender<JsonValue>;

/// Runtime context for graph execution
#[derive(Clone)]
pub struct Runtime<Ctx: Clone = ()> {
    /// Run-scoped immutable context (user_id, db_conn, etc.)
    pub context: Ctx,
    /// Persistence/memory store
    pub store: Option<Arc<dyn BaseStore>>,
    /// Channel sender for custom streaming
    pub stream_writer: Option<StreamWriter>,
    /// Previous return value (functional API)
    pub previous: Option<JsonValue>,
    /// Execution metadata
    pub execution_info: Option<ExecutionInfo>,
    /// Server metadata
    pub server_info: Option<ServerInfo>,
}

/// Execution info for the current task
#[derive(Debug, Clone)]
pub struct ExecutionInfo {
    pub checkpoint_id: String,
    pub checkpoint_ns: String,
    pub task_id: String,
    pub thread_id: Option<String>,
    pub run_id: Option<String>,
}

/// Server info from LangGraph Server
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub server_url: Option<String>,
}
