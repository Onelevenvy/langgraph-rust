//! Prebuilt components for LangGraph agents.
//!
//! This crate provides prebuilt components for common agent patterns:
//!
//! - **Message types**: `Message`, `ToolCall`, `MessageContent`
//! - **Traits**: `BaseTool`, `BaseChatModel` for tool and model integration
//! - **ToolNode**: Executes tool calls from AI responses
//! - **create_react_agent**: Builds a ReAct (Reasoning + Acting) agent graph
//! - **tools_condition**: Routing function for tool-calling agents

pub mod types;
pub mod traits;
pub mod tool_node;
pub mod chat_agent;
pub mod tools_condition;
pub mod node_helpers;

pub use types::{Message, ToolCall, MessageContent, ContentBlock, add_messages, add_messages_ref};
pub use traits::{BaseTool, BaseChatModel, MessageStream, ToolDef, ClosureTool, ToolError, ModelError, PreparedTools, prepare_tools, LlmUsage};
pub use tool_node::ToolNode;
pub use chat_agent::{create_react_agent, ReActAgent, ReActAgentConfig};
pub use tools_condition::tools_condition;
pub use node_helpers::{extract_messages, llm_response_to_json, invoke_llm, invoke_llm_with_config, stream_llm, get_i64, get_str, response_text, parse_json_response, ask_json, stream_and_print};
