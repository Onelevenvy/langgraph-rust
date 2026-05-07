use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph::runnable::{Runnable, RunnableError};

use crate::traits::{BaseTool, ToolError};
use crate::types::{Message, ToolCall};

/// Result of executing a tool call.
enum ToolCallResult {
    /// Normal tool message.
    Message(Message),
    /// A Command returned by the tool (for state updates, resume, goto).
    Command {
        /// The tool_call_id for this invocation.
        tool_call_id: String,
        /// Extra messages from the Command.update (e.g., ToolMessages with state updates).
        extra_messages: Vec<JsonValue>,
        /// State update fields from Command.update (excluding messages).
        state_update: serde_json::Map<String, JsonValue>,
    },
}

/// Error message templates for tool invocation failures.
const INVALID_TOOL_NAME_ERROR: &str = "Error: {requested_tool} is not a valid tool, try one of [{available_tools}].";
const TOOL_CALL_ERROR: &str = "Error: {error}\n Please fix your mistakes.";
const TOOL_EXECUTION_ERROR: &str = "Error executing tool '{tool_name}' with kwargs {tool_kwargs} with error:\n {error}\n Please fix the error and try again.";

/// A node that executes tool calls from the AI's response.
///
/// ToolNode reads tool calls from the last AI message and executes them
/// in parallel, returning the results as tool messages.
pub struct ToolNode {
    tools: HashMap<String, Arc<dyn BaseTool>>,
    handle_tool_errors: bool,
}

impl ToolNode {
    /// Create a new ToolNode with the given tools.
    pub fn new(tools: Vec<Arc<dyn BaseTool>>) -> Self {
        let tool_map: HashMap<String, Arc<dyn BaseTool>> = tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();

        Self {
            tools: tool_map,
            handle_tool_errors: true,
        }
    }

    /// Set whether to handle tool errors gracefully (returning error messages
    /// instead of propagating).
    pub fn with_error_handling(mut self, handle: bool) -> Self {
        self.handle_tool_errors = handle;
        self
    }

    /// Get the list of available tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Extract tool calls from the input state.
    /// Expects a JSON object with a "messages" array containing AI messages with tool calls.
    fn extract_tool_calls(input: &JsonValue) -> Vec<ToolCall> {
        let messages = match input.get("messages") {
            Some(JsonValue::Array(arr)) => arr,
            _ => return vec![],
        };

        // Get the last AI message with tool calls
        for msg in messages.iter().rev() {
            if let Some(obj) = msg.as_object() {
                if obj.get("type").and_then(|v| v.as_str()) == Some("ai") {
                    if let Some(JsonValue::Array(calls)) = obj.get("tool_calls") {
                        return calls
                            .iter()
                            .filter_map(|tc| serde_json::from_value(tc.clone()).ok())
                            .collect();
                    }
                }
            }
        }

        vec![]
    }
}

#[async_trait]
impl Runnable for ToolNode {
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        // Use tokio runtime for sync invocation
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.ainvoke(input, config)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| RunnableError::Node(e.to_string()))?;
                rt.block_on(self.ainvoke(input, config))
            }
        }
    }

    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let tool_calls = Self::extract_tool_calls(input);

        if tool_calls.is_empty() {
            return Ok(serde_json::json!({}));
        }

        // Execute all tool calls concurrently using JoinSet
        let mut join_set = tokio::task::JoinSet::new();
        for tc in tool_calls {
            let tool = self.tools.get(&tc.name).cloned();
            let config = config.clone();
            let handle_errors = self.handle_tool_errors;
            let tool_name = tc.name.clone();
            let available_tools: Vec<String> = self.tools.keys().cloned().collect();

            join_set.spawn(async move {
                let tool = match tool {
                    Some(t) => t,
                    None => {
                        return Err(ToolError::NotFound(
                            INVALID_TOOL_NAME_ERROR
                                .replace("{requested_tool}", &tc.name)
                                .replace("{available_tools}", &available_tools.join(", ")),
                        ));
                    }
                };

                let result = tool.ainvoke(&tc.args, &config).await;
                let tool_call_id = tc.id.clone().unwrap_or_default();

                match result {
                    Ok(output) => {
                        // If the output is a string, try to parse it as JSON
                        // (handles tools that return serialized JSON strings)
                        let output = match &output {
                            JsonValue::String(s) => serde_json::from_str(s).unwrap_or(output),
                            _ => output,
                        };

                        // Check if the tool returned a Command (has "update" or "resume" field)
                        if let Some(obj) = output.as_object() {
                            if obj.contains_key("update") || obj.contains_key("resume") {
                                let mut state_update = serde_json::Map::new();
                                let mut extra_messages: Vec<JsonValue> = Vec::new();

                                if let Some(update) = obj.get("update") {
                                    if let Some(update_obj) = update.as_object() {
                                        // Extract messages from update, fix up tool_call_id
                                        if let Some(JsonValue::Array(msgs)) = update_obj.get("messages") {
                                            for msg in msgs {
                                                let mut msg = msg.clone();
                                                // Fix up tool_call_id in each message
                                                if let Some(msg_obj) = msg.as_object_mut() {
                                                    if msg_obj.contains_key("tool_call_id") {
                                                        msg_obj.insert(
                                                            "tool_call_id".to_string(),
                                                            JsonValue::String(tool_call_id.clone()),
                                                        );
                                                    }
                                                }
                                                extra_messages.push(msg);
                                            }
                                        }
                                        // Collect non-messages fields as state updates
                                        for (k, v) in update_obj {
                                            if k != "messages" {
                                                state_update.insert(k.clone(), v.clone());
                                            }
                                        }
                                    }
                                }

                                return Ok(ToolCallResult::Command {
                                    tool_call_id,
                                    extra_messages,
                                    state_update,
                                });
                            }
                        }

                        let content = match output {
                            JsonValue::String(s) => s,
                            other => serde_json::to_string_pretty(&other).unwrap_or_else(|_| format!("{:?}", other)),
                        };
                        Ok(ToolCallResult::Message(Message::tool_result(tool_call_id, content)))
                    }
                    Err(crate::traits::ToolError::Interrupt(interrupt)) => {
                        Err(crate::traits::ToolError::Interrupt(interrupt))
                    }
                    Err(e) => {
                        if handle_errors {
                            let error_msg = TOOL_EXECUTION_ERROR
                                .replace("{tool_name}", &tool_name)
                                .replace("{tool_kwargs}", &serde_json::to_string(&tc.args).unwrap_or_default())
                                .replace("{error}", &e.to_string());
                            Ok(ToolCallResult::Message(Message::tool_error(tool_call_id, error_msg)))
                        } else {
                            Err(e)
                        }
                    }
                }
            });
        }

        // Collect results from all spawned tasks
        let mut messages: Vec<JsonValue> = Vec::new();
        let mut state_updates: serde_json::Map<String, JsonValue> = serde_json::Map::new();

        while let Some(result) = join_set.join_next().await {
            let msg_result = result.map_err(|e| RunnableError::Node(e.to_string()))?;
            match msg_result {
                Ok(ToolCallResult::Message(msg)) => {
                    messages.push(serde_json::to_value(msg).map_err(|e| RunnableError::Node(e.to_string()))?);
                }
                Ok(ToolCallResult::Command { tool_call_id, extra_messages, state_update }) => {
                    if extra_messages.is_empty() {
                        // No messages in Command — add a default tool response
                        let default_msg = Message::tool_result(tool_call_id, "Command processed");
                        messages.push(serde_json::to_value(default_msg).map_err(|e| RunnableError::Node(e.to_string()))?);
                    } else {
                        messages.extend(extra_messages);
                    }
                    // Merge state updates from Command
                    for (k, v) in state_update {
                        state_updates.insert(k, v);
                    }
                }
                Err(ToolError::Interrupt(interrupt)) => {
                    return Err(RunnableError::Interrupt(interrupt));
                }
                Err(e) => {
                    return Err(RunnableError::Node(e.to_string()));
                }
            }
        }

        // Build result: messages + any state updates from Commands
        let mut result = serde_json::json!({ "messages": messages });
        if let Some(obj) = result.as_object_mut() {
            for (k, v) in state_updates {
                obj.insert(k, v);
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "ToolNode"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_extract_tool_calls() {
        let input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "Search for cats"},
                {"type": "ai", "content": "", "tool_calls": [
                    {"name": "search", "args": {"query": "cats"}, "id": "call_1"}
                ]}
            ]
        });

        let calls = ToolNode::extract_tool_calls(&input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_extract_no_tool_calls() {
        let input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"}
            ]
        });

        let calls = ToolNode::extract_tool_calls(&input);
        assert!(calls.is_empty());
    }
}
