use std::pin::Pin;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph::types::{GraphInterrupt, InterruptError};
use crate::types::Message;

/// A stream of message chunks from a chat model.
///
/// Each item is a `Message` representing either a partial token chunk
/// (for real-time display) or the final complete message.
pub type MessageStream<'a> = Pin<Box<dyn tokio_stream::Stream<Item = Result<Message, ModelError>> + Send + 'a>>;

/// Error type for tool and model operations.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool execution error: {0}")]
    Execution(String),

    #[error("invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("tool not found: {0}")]
    NotFound(String),

    #[error("graph interrupt")]
    Interrupt(GraphInterrupt),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        ToolError::Execution(s)
    }
}

impl From<GraphInterrupt> for ToolError {
    fn from(interrupt: GraphInterrupt) -> Self {
        ToolError::Interrupt(interrupt)
    }
}

impl From<InterruptError> for ToolError {
    fn from(e: InterruptError) -> Self {
        ToolError::Interrupt(e.into())
    }
}

/// Error type for chat model operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("model invocation error: {0}")]
    Invocation(String),

    #[error("model configuration error: {0}")]
    Config(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// A tool that can be invoked by an agent.
///
/// Mirrors langchain-core's BaseTool.
#[async_trait]
pub trait BaseTool: Send + Sync {
    /// The name of the tool.
    fn name(&self) -> &str;

    /// A description of what the tool does.
    fn description(&self) -> &str {
        ""
    }

    /// The JSON schema for the tool's parameters.
    fn parameters(&self) -> Option<&JsonValue> {
        None
    }

    /// Invoke the tool synchronously with the given arguments.
    fn invoke(&self, args: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, ToolError>;

    /// Invoke the tool asynchronously. Default delegates to sync invoke via block_in_place.
    ///
    /// Sets up thread-local config/runtime context so that `get_config()` and
    /// `get_runtime()` work inside sync tool code (needed by `interrupt()`).
    async fn ainvoke(&self, args: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, ToolError> {
        let args = args.clone();
        let config = config.clone();
        // Capture runtime from async task-locals if available
        let current_runtime = langgraph::config::get_runtime();
        // Always use with_runtime_sync to set up thread-locals for get_config()/get_runtime()
        let runtime = current_runtime.unwrap_or_else(|| {
            std::sync::Arc::new(langgraph::runtime::Runtime {
                context: (),
                store: None,
                stream_writer: None,
                previous: None,
                execution_info: None,
                server_info: None,
            })
        });
        tokio::task::block_in_place(|| {
            langgraph::config::with_runtime_sync(config.clone(), runtime, || {
                self.invoke(&args, &config)
            })
        })
    }

    /// Get the tool's schema as a ToolCall-compatible descriptor.
    fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters().cloned().unwrap_or(serde_json::json!({})),
        }
    }
}

/// A tool definition that can be passed to a chat model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: JsonValue,
}

/// A chat model that can generate responses.
///
/// Mirrors langchain-core's BaseChatModel.
#[async_trait]
pub trait BaseChatModel: Send + Sync {
    /// The name of the model.
    fn name(&self) -> &str;

    /// Invoke the model with a list of messages and get a response.
    fn invoke(&self, messages: &[Message], config: &RunnableConfig) -> Result<Message, ModelError>;

    /// Invoke the model asynchronously. Default delegates to sync invoke via block_in_place.
    async fn ainvoke(&self, messages: &[Message], config: &RunnableConfig) -> Result<Message, ModelError> {
        let messages = messages.to_vec();
        let config = config.clone();
        tokio::task::block_in_place(|| self.invoke(&messages, &config))
    }

    /// Stream tokens from the model. Returns a stream of partial Message chunks.
    ///
    /// Each yielded `Message` represents the accumulated content up to that point.
    /// For example, if the model generates "Hello world", the stream might yield:
    /// - `Message::ai("Hello")`
    /// - `Message::ai("Hello world")`
    ///
    /// The final item in the stream is the complete response (including tool calls if any).
    ///
    /// Default implementation falls back to `ainvoke` (yields a single complete message).
    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        let messages = messages.to_vec();
        let config = config.clone();
        Box::pin(async_stream::stream! {
            let msg = self.ainvoke(&messages, &config).await?;
            yield Ok(msg);
        })
    }

    /// Bind tools to the model for tool-calling support.
    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel>;
}

/// A simple tool implemented as a closure.
pub struct ClosureTool {
    tool_name: String,
    tool_description: String,
    tool_parameters: Option<JsonValue>,
    func: Box<dyn Fn(&JsonValue) -> Result<JsonValue, ToolError> + Send + Sync>,
}

impl ClosureTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        func: impl Fn(&JsonValue) -> Result<JsonValue, ToolError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            tool_name: name.into(),
            tool_description: description.into(),
            tool_parameters: None,
            func: Box::new(func),
        }
    }

    pub fn with_parameters(mut self, params: JsonValue) -> Self {
        self.tool_parameters = Some(params);
        self
    }
}

#[async_trait]
impl BaseTool for ClosureTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn parameters(&self) -> Option<&JsonValue> {
        self.tool_parameters.as_ref()
    }

    fn invoke(&self, args: &JsonValue, _config: &RunnableConfig) -> Result<JsonValue, ToolError> {
        (self.func)(args)
    }
}

/// Result of `prepare_tools()`: contains everything you need to work with tools.
///
/// # Fields
/// - `tool_defs`: Tool definitions for binding to a model (`model.bind_tools(prepared.tool_defs)`)
/// - `by_name`: Name-to-tool lookup map for executing tool calls
/// - `tools`: The original tools list (for passing to `ToolNode`, etc.)
pub struct PreparedTools {
    pub tool_defs: Vec<ToolDef>,
    pub by_name: std::collections::HashMap<String, std::sync::Arc<dyn BaseTool>>,
    pub tools: Vec<std::sync::Arc<dyn BaseTool>>,
}

/// Prepare tools for use in a graph.
///
/// Takes a list of tools and returns everything needed:
/// - `tool_defs`: for `model.bind_tools()`
/// - `by_name`: for looking up tools by name when executing calls
/// - `tools`: original list for `ToolNode` or other uses
///
/// # Example
/// ```ignore
/// use langgraph_prebuilt::prepare_tools;
///
/// let prepared = prepare_tools(vec![
///     Arc::new(Multiply::new()),
///     Arc::new(Add::new()),
/// ]);
///
/// let model = model.bind_tools(prepared.tool_defs);
/// // Use prepared.by_name in tool_node closure
/// ```
pub fn prepare_tools(tools: Vec<std::sync::Arc<dyn BaseTool>>) -> PreparedTools {
    let tool_defs: Vec<ToolDef> = tools.iter().map(|t| t.to_tool_def()).collect();
    let by_name: std::collections::HashMap<String, std::sync::Arc<dyn BaseTool>> = tools
        .iter()
        .map(|t| (t.name().to_string(), t.clone()))
        .collect();
    PreparedTools {
        tool_defs,
        by_name,
        tools,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_tool() {
        let tool = ClosureTool::new("echo", "Echoes the input", |args| {
            Ok(args.clone())
        });

        assert_eq!(tool.name(), "echo");
        assert_eq!(tool.description(), "Echoes the input");

        let result = tool.invoke(&serde_json::json!("hello"), &RunnableConfig::new()).unwrap();
        assert_eq!(result, serde_json::json!("hello"));
    }
}
