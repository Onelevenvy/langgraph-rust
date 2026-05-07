use serde_json::Value as JsonValue;
use langgraph::constants::END;

/// Routing function for determining whether to continue with tool execution
/// or end the conversation.
///
/// This is a standard conditional edge function for ReAct-style agents.
/// It checks if the last AI message contains tool calls:
/// - If yes, route to the "tools" node
/// - If no, route to END
///
/// Returns a string key that should be mapped in the conditional edges:
/// - "tools" → route to ToolNode
/// - END ("__end__") → route to END
pub fn tools_condition(input: &JsonValue) -> String {
    let messages = match input.get("messages") {
        Some(JsonValue::Array(arr)) => arr,
        _ => return END.to_string(),
    };

    // Check the last message for tool calls
    if let Some(last_msg) = messages.last() {
        if let Some(obj) = last_msg.as_object() {
            if obj.get("type").and_then(|v| v.as_str()) == Some("ai") {
                if let Some(JsonValue::Array(calls)) = obj.get("tool_calls") {
                    if !calls.is_empty() {
                        return "tools".to_string();
                    }
                }
            }
        }
    }

    END.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tools_condition_with_tool_calls() {
        let input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "Search for cats"},
                {"type": "ai", "content": "", "tool_calls": [
                    {"name": "search", "args": {"query": "cats"}, "id": "call_1"}
                ]}
            ]
        });

        assert_eq!(tools_condition(&input), "tools");
    }

    #[test]
    fn test_tools_condition_without_tool_calls() {
        let input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "Hello"},
                {"type": "ai", "content": "Hi there!"}
            ]
        });

        assert_eq!(tools_condition(&input), "__end__");
    }

    #[test]
    fn test_tools_condition_empty_messages() {
        let input = serde_json::json!({
            "messages": []
        });

        assert_eq!(tools_condition(&input), "__end__");
    }

    #[test]
    fn test_tools_condition_no_messages_key() {
        let input = serde_json::json!({});
        assert_eq!(tools_condition(&input), "__end__");
    }
}
