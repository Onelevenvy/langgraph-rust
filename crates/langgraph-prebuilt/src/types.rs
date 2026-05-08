use std::fmt;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

fn default_tool_status() -> String {
    "success".to_string()
}

/// A tool call requested by the AI model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The name of the tool to call.
    pub name: String,
    /// The arguments to pass to the tool, as a JSON object.
    pub args: JsonValue,
    /// A unique identifier for this tool call.
    #[serde(default)]
    pub id: Option<String>,
}

/// Message types for the agent system.
///
/// Mirrors the LangChain message types: HumanMessage, AIMessage,
/// SystemMessage, ToolMessage, and RemoveMessage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    /// A message from the human user.
    Human {
        content: MessageContent,
        #[serde(default)]
        id: Option<String>,
    },
    /// A message from the AI assistant.
    Ai {
        content: MessageContent,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        id: Option<String>,
        /// Token usage from the LLM API response, if available.
        #[serde(default)]
        usage: Option<crate::traits::LlmUsage>,
    },
    /// A system message providing instructions.
    System {
        content: MessageContent,
        #[serde(default)]
        id: Option<String>,
    },
    /// A message containing the result of a tool call.
    Tool {
        content: MessageContent,
        tool_call_id: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        id: Option<String>,
        /// Status of the tool call: "success" or "error"
        #[serde(default = "default_tool_status")]
        status: String,
    },
    /// A message that removes a previous message by ID.
    Remove {
        id: String,
    },
}

/// Content of a message - can be a simple string or structured content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Structured content blocks (for multimodal messages).
    Blocks(Vec<ContentBlock>),
}

/// A block of content within a message (text, image, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
}

/// An image URL reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

impl Message {
    /// Get the text content of the message, if any.
    pub fn text(&self) -> Option<&str> {
        match self {
            Message::Human { content, .. }
            | Message::Ai { content, .. }
            | Message::System { content, .. }
            | Message::Tool { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Blocks(blocks) => {
                    // Return the first text block
                    blocks.iter().find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                }
            },
            Message::Remove { .. } => None,
        }
    }

    /// Get the message ID, if any.
    pub fn id(&self) -> Option<&str> {
        match self {
            Message::Human { id, .. }
            | Message::Ai { id, .. }
            | Message::System { id, .. }
            | Message::Tool { id, .. } => id.as_deref(),
            Message::Remove { id } => Some(id.as_str()),
        }
    }

    /// Check if this message has tool calls.
    pub fn has_tool_calls(&self) -> bool {
        match self {
            Message::Ai { tool_calls, .. } => !tool_calls.is_empty(),
            _ => false,
        }
    }

    /// Get tool calls from the message.
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Message::Ai { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }

    /// Create a human message.
    pub fn human(content: impl Into<String>) -> Self {
        Message::Human {
            content: MessageContent::Text(content.into()),
            id: None,
        }
    }

    /// Create an AI message.
    pub fn ai(content: impl Into<String>) -> Self {
        Message::Ai {
            content: MessageContent::Text(content.into()),
            tool_calls: vec![],
            id: None,
            usage: None,
        }
    }

    /// Create an AI message with tool calls.
    pub fn ai_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Message::Ai {
            content: MessageContent::Text(content.into()),
            tool_calls,
            id: None,
            usage: None,
        }
    }

    /// Create an AI message with token usage information.
    pub fn ai_with_usage(content: impl Into<String>, usage: crate::traits::LlmUsage) -> Self {
        Message::Ai {
            content: MessageContent::Text(content.into()),
            tool_calls: vec![],
            id: None,
            usage: Some(usage),
        }
    }

    /// Create an AI message with tool calls and token usage.
    pub fn ai_with_tool_calls_and_usage(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
        usage: crate::traits::LlmUsage,
    ) -> Self {
        Message::Ai {
            content: MessageContent::Text(content.into()),
            tool_calls,
            id: None,
            usage: Some(usage),
        }
    }

    /// Get token usage from the message, if available.
    pub fn usage(&self) -> Option<&crate::traits::LlmUsage> {
        match self {
            Message::Ai { usage, .. } => usage.as_ref(),
            _ => None,
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Message::System {
            content: MessageContent::Text(content.into()),
            id: None,
        }
    }

    /// Create a tool result message with success status.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool {
            content: MessageContent::Text(content.into()),
            tool_call_id: tool_call_id.into(),
            name: None,
            id: None,
            status: "success".to_string(),
        }
    }

    /// Create a tool error message.
    pub fn tool_error(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message::Tool {
            content: MessageContent::Text(content.into()),
            tool_call_id: tool_call_id.into(),
            name: None,
            id: None,
            status: "error".to_string(),
        }
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Message::Human { content, .. } => write!(f, "[Human] {}", content_text(content)),
            Message::Ai { content, tool_calls, .. } => {
                let text = content_text(content);
                if tool_calls.is_empty() {
                    write!(f, "[AI] {}", text)
                } else {
                    let calls: Vec<String> = tool_calls
                        .iter()
                        .map(|tc| format!("{}({})", tc.name, tc.args))
                        .collect();
                    if text.is_empty() {
                        write!(f, "[AI] → {}", calls.join(", "))
                    } else {
                        write!(f, "[AI] {} → {}", text, calls.join(", "))
                    }
                }
            }
            Message::System { content, .. } => write!(f, "[System] {}", content_text(content)),
            Message::Tool { content, name, status, .. } => {
                let tool_name = name.as_deref().unwrap_or("tool");
                let text = content_text(content);
                if status == "error" {
                    write!(f, "[Tool:{}] ERROR: {}", tool_name, text)
                } else {
                    write!(f, "[Tool:{}] {}", tool_name, text)
                }
            }
            Message::Remove { id } => write!(f, "[Remove:{}]", id),
        }
    }
}

fn content_text(content: &MessageContent) -> &str {
    match content {
        MessageContent::Text(s) => s.as_str(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or(""),
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// Merge function for messages: appends new messages to existing ones.
/// This is the default reducer for the `messages` field in agent states.
pub fn add_messages(current: JsonValue, update: JsonValue) -> JsonValue {
    let messages: Vec<JsonValue> = match current {
        JsonValue::Array(arr) => arr,
        _ => vec![],
    };

    let new_messages: Vec<JsonValue> = match update {
        JsonValue::Array(arr) => arr,
        other => vec![other],
    };

    // Handle RemoveMessage by filtering out messages with matching IDs
    let mut result: Vec<JsonValue> = Vec::new();
    let mut remove_ids: Vec<String> = Vec::new();

    // Collect IDs to remove
    for msg in &new_messages {
        if let Some(obj) = msg.as_object() {
            if obj.get("type").and_then(|v| v.as_str()) == Some("remove") {
                if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
                    remove_ids.push(id.to_string());
                }
            }
        }
    }

    // Add existing messages, skipping removed ones
    for msg in messages {
        if let Some(id) = msg.get("id").and_then(|v| v.as_str()) {
            if remove_ids.contains(&id.to_string()) {
                continue;
            }
        }
        result.push(msg);
    }

    // Add new non-remove messages
    for msg in new_messages {
        if let Some(obj) = msg.as_object() {
            if obj.get("type").and_then(|v| v.as_str()) == Some("remove") {
                continue;
            }
        }
        result.push(msg);
    }

    JsonValue::Array(result)
}

/// Merge function for messages with reference signature.
///
/// This is the version compatible with `#[channel(reducer = "...")]` in the
/// derive macro, which expects `fn(&JsonValue, &JsonValue) -> JsonValue`.
///
/// ```ignore
/// #[derive(StateGraph)]
/// struct MyState {
///     #[channel(reducer = "add_messages_ref")]
///     messages: Vec<Message>,
/// }
/// ```
pub fn add_messages_ref(current: &JsonValue, update: &JsonValue) -> JsonValue {
    add_messages(current.clone(), update.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_human() {
        let msg = Message::human("Hello");
        assert_eq!(msg.text(), Some("Hello"));
        assert!(msg.id().is_none());
    }

    #[test]
    fn test_message_ai_with_tool_calls() {
        let tc = ToolCall {
            name: "search".into(),
            args: serde_json::json!({"query": "test"}),
            id: Some("call_1".into()),
        };
        let msg = Message::ai_with_tool_calls("", vec![tc]);
        assert!(msg.has_tool_calls());
        assert_eq!(msg.tool_calls().len(), 1);
        assert_eq!(msg.tool_calls()[0].name, "search");
    }

    #[test]
    fn test_add_messages() {
        let existing = serde_json::json!([
            {"type": "human", "content": "Hi"},
        ]);
        let update = serde_json::json!([
            {"type": "ai", "content": "Hello"},
        ]);
        let result = add_messages(existing, update);
        assert_eq!(result.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_remove_message() {
        let existing = serde_json::json!([
            {"type": "human", "content": "Hi", "id": "msg1"},
            {"type": "ai", "content": "Hello", "id": "msg2"},
        ]);
        let update = serde_json::json!([
            {"type": "remove", "id": "msg1"},
        ]);
        let result = add_messages(existing, update);
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "msg2");
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::human("Hello world");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("human"));
        assert!(json.contains("Hello world"));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.text(), Some("Hello world"));
    }
}
