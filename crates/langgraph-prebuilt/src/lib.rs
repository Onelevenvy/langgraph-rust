//! Prebuilt components for LangGraph agents.
//!
//! This crate provides prebuilt components for common agent patterns:
//!
//! - **Message types**: `Message`, `ToolCall`, `MessageContent`
//! - **Traits**: `BaseTool`, `BaseChatModel` for tool and model integration
//! - **ToolNode**: Executes tool calls from AI responses
//! - **create_react_agent**: Builds a ReAct (Reasoning + Acting) agent graph
//! - **tools_condition**: Routing function for tool-calling agents

pub mod chat_agent;
pub mod node_helpers;
pub mod tool_node;
pub mod tools_condition;
pub mod traits;
pub mod types;

pub use chat_agent::{create_react_agent, ReActAgent, ReActAgentConfig};
pub use node_helpers::{
    ask_json, extract_messages, get_i64, get_str, invoke_llm, invoke_llm_with_config,
    llm_response_to_json, parse_json_response, print_result, print_result_with_options,
    print_stream, print_stream_with_options, response_text, stream_llm,
};
pub use tool_node::ToolNode;
pub use tools_condition::tools_condition;
pub use traits::{
    prepare_tools, BaseChatModel, BaseTool, ClosureTool, LlmUsage, MessageStream, ModelError,
    PreparedTools, ToolDef, ToolError,
};
pub use types::{
    add_messages, add_messages_ref, ContentBlock, ImageUrl, Message, MessageContent, ToolCall,
};
