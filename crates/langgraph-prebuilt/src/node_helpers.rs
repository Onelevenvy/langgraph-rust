//! Helper functions for building graph nodes with minimal boilerplate.
//!
//! These utilities eliminate the manual JSON ↔ typed conversion that makes
//! Rust examples verbose compared to Python's langchain-core.

use serde_json::Value as JsonValue;
use tokio_stream::StreamExt;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph::config::get_stream_writer;
use langgraph::runnable::RunnableError;

use crate::traits::BaseChatModel;
use crate::types::Message;

/// Extract typed messages from a graph state JSON, with an optional system prompt prepended.
///
/// This replaces the common 8-line pattern:
/// ```ignore
/// let messages_json = input.get("messages")
///     .and_then(|m| m.as_array()).cloned().unwrap_or_default();
/// let mut typed_messages = vec![Message::system("...")];
/// for msg in &messages_json {
///     if let Ok(m) = serde_json::from_value::<Message>(msg.clone()) {
///         typed_messages.push(m);
///     }
/// }
/// ```
///
/// With:
/// ```ignore
/// let messages = extract_messages(&input, Some("You are a helpful assistant."));
/// ```
pub fn extract_messages(input: &JsonValue, system_prompt: Option<&str>) -> Vec<Message> {
    let messages_json = input
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    let mut messages = Vec::with_capacity(messages_json.len() + 1);

    if let Some(prompt) = system_prompt {
        messages.push(Message::system(prompt));
    }

    for msg in &messages_json {
        if let Ok(m) = serde_json::from_value::<Message>(msg.clone()) {
            messages.push(m);
        }
    }

    messages
}

/// Convert a model response into a state update JSON.
///
/// Wraps the response message in `{"messages": [response]}` format.
pub fn llm_response_to_json(response: Message) -> Result<JsonValue, RunnableError> {
    let response_json = serde_json::to_value(response)
        .map_err(|e| RunnableError::Node(e.to_string()))?;
    Ok(serde_json::json!({ "messages": [response_json] }))
}

/// Invoke an LLM and return a state update.
///
/// This is the complete LLM node logic in one call:
/// 1. Extracts messages from input state
/// 2. Prepends system prompt
/// 3. Calls the model
/// 4. Wraps response in state update format
///
/// # Example
/// ```ignore
/// let model_clone = model.clone();
/// graph.add_node("chatbot", move |input: JsonValue, _config: RunnableConfig| {
///     let model = model_clone.clone();
///     async move { invoke_llm(model.as_ref(), &input, "You are a helpful assistant.") }
/// })?;
/// ```
pub fn invoke_llm(
    model: &dyn BaseChatModel,
    input: &JsonValue,
    system_prompt: &str,
) -> Result<JsonValue, RunnableError> {
    let messages = extract_messages(input, Some(system_prompt));
    let response = model.invoke(&messages, &RunnableConfig::new())
        .map_err(|e| RunnableError::Node(e.to_string()))?;
    llm_response_to_json(response)
}

/// Invoke an LLM with a custom config and return a state update.
///
/// Same as [`invoke_llm`] but allows passing a custom config (e.g., for streaming).
pub fn invoke_llm_with_config(
    model: &dyn BaseChatModel,
    input: &JsonValue,
    system_prompt: &str,
    config: &RunnableConfig,
) -> Result<JsonValue, RunnableError> {
    let messages = extract_messages(input, Some(system_prompt));
    let response = model.invoke(messages.as_slice(), config)
        .map_err(|e| RunnableError::Node(e.to_string()))?;
    llm_response_to_json(response)
}

/// Stream LLM tokens via StreamWriter and return the final state update.
///
/// Calls `model.astream()` for token-by-token streaming. Each partial message
/// is forwarded through the stream writer (if active) as a JSON payload:
/// ```json
/// {"type": "token", "content": "Hello"}
/// ```
///
/// The final complete message is returned as a state update in
/// `{"messages": [response]}` format.
///
/// # Example
/// ```ignore
/// let model_clone = model.clone();
/// graph.add_node("chatbot", move |input: JsonValue, _config: RunnableConfig| {
///     let model = model_clone.clone();
///     async move { stream_llm(model.as_ref(), &input, "You are a helpful assistant.").await }
/// })?;
/// ```
pub async fn stream_llm(
    model: &(dyn BaseChatModel + Send + Sync),
    input: &JsonValue,
    system_prompt: &str,
) -> Result<JsonValue, RunnableError> {
    let messages = extract_messages(input, Some(system_prompt));
    let writer = get_stream_writer();

    let config = RunnableConfig::new();
    let mut stream = model.astream(&messages, &config);
    let mut accumulated = String::new();
    let mut last_message = None;

    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| RunnableError::Node(e.to_string()))?;

        // Forward delta content via StreamWriter for real-time display
        if let Some(ref w) = writer {
            if let Some(content) = chunk.text() {
                if !content.is_empty() {
                    let _ = w.try_send(serde_json::json!({
                        "type": "token",
                        "content": content,
                    }));
                }
            }
        }

        // Accumulate text from delta chunks
        if let Some(text) = chunk.text() {
            accumulated.push_str(text);
        }
        last_message = Some(chunk);
    }

    // Use last chunk if it carries tool calls (e.g. the model called tools
    // without generating text content); otherwise build from accumulated text.
    let final_message = match last_message {
        Some(msg) if msg.has_tool_calls() => msg,
        _ => Message::ai(accumulated),
    };
    llm_response_to_json(final_message)
}

/// Get a field from state as i64, defaulting to 0.
pub fn get_i64(input: &JsonValue, key: &str) -> i64 {
    input.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
}

/// Get a field from state as a string, defaulting to "".
pub fn get_str<'a>(input: &'a JsonValue, key: &str) -> &'a str {
    input.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

/// Extract the assistant's text reply from an `invoke_llm` / `stream_llm` result.
///
/// Both helpers return `{"messages": [response]}`. This function digs out the
/// `content` field of the last message so callers don't repeat the same
/// `.get("messages") … .last() … .get("content")` chain every time.
///
/// # Example
/// ```ignore
/// let result = stream_llm(model, &input, "You are a planner.").await?;
/// let text = response_text(&result);
/// println!("LLM said: {}", text);
/// ```
pub fn response_text(result: &JsonValue) -> &str {
    result
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
}

/// Strip markdown code fences (` ```json … ``` `) and parse the inner JSON.
///
/// If the text is plain JSON (no fences), it is parsed directly.
/// Returns `None` when the text is not valid JSON after stripping.
///
/// # Example
/// ```ignore
/// let text = r#"```json\n{"title": "Plan"}\n```"#;
/// let value = parse_json_response(text).unwrap();
/// assert_eq!(value["title"], "Plan");
/// ```
pub fn parse_json_response(text: &str) -> Option<JsonValue> {
    let trimmed = text.trim();
    let json_str = if trimmed.starts_with("```") {
        let start = trimmed.find('\n').map(|i| i + 1).unwrap_or(3);
        let end = trimmed.rfind("```").unwrap_or(trimmed.len());
        &trimmed[start..end]
    } else {
        trimmed
    };
    serde_json::from_str(json_str.trim()).ok()
}

/// Ask the LLM a single prompt and get back a parsed JSON value.
///
/// This combines three steps that are repeated in every "structured output" node:
/// 1. Call `stream_llm` with a raw prompt (no state extraction)
/// 2. Extract the response text
/// 3. Parse JSON (stripping markdown fences if present)
///
/// Returns `None` when the response is not valid JSON.
///
/// # Example
/// ```ignore
/// let plan = ask_json(model, "Create a plan in JSON format", "").await;
/// ```
pub async fn ask_json(
    model: &(dyn BaseChatModel + Send + Sync),
    prompt: &str,
    system_prompt: &str,
) -> Result<Option<JsonValue>, RunnableError> {
    let input = serde_json::json!({"messages": [{"type": "human", "content": prompt}]});
    let result = stream_llm(model, &input, system_prompt).await?;
    let text = response_text(&result);
    Ok(parse_json_response(text))
}
