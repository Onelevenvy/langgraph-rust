//! Helper functions for building graph nodes with minimal boilerplate.
//!
//! These utilities eliminate the manual JSON ↔ typed conversion that makes
//! Rust examples verbose compared to Python's langchain-core.

use langgraph::config::get_stream_writer;
use langgraph::runnable::RunnableError;
use langgraph::stream::StreamPart;
use langgraph::types::StreamMode;
use langgraph_checkpoint::config::RunnableConfig;
use serde_json::Value as JsonValue;
use std::io::Write;
use tokio_stream::StreamExt;

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
    let response_json =
        serde_json::to_value(response).map_err(|e| RunnableError::Node(e.to_string()))?;
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
    let response = model
        .invoke(&messages, &RunnableConfig::new())
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
    let response = model
        .invoke(messages.as_slice(), config)
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
    let mut accumulated_thinking = String::new();
    let mut accumulated_content = String::new();
    let mut tool_calls_message = None;

    // Standard incremental streaming (same as LangChain / OpenAI SDK):
    //   - Each chunk yielded by the provider contains ONLY new delta tokens.
    //   - If tool calls are present, the provider yields ONE final signal chunk
    //     at the very end with has_tool_calls()=true and empty content/thinking.
    //     This lets us detect tool calls without re-printing any content.
    //   - We forward every content/thinking delta directly to the stream writer,
    //     and accumulate them ourselves for the final return value.
    while let Some(result) = stream.next().await {
        let chunk = result.map_err(|e| RunnableError::Node(e.to_string()))?;

        if chunk.has_tool_calls() {
            // Tool-calls signal chunk — no content to print, just capture it.
            tool_calls_message = Some(chunk);
        } else {
            // Pure delta chunk — forward to stream writer and accumulate.
            if let Some(ref w) = writer {
                if let Some(thinking) = chunk.thinking() {
                    if !thinking.is_empty() {
                        let _ = w.try_send(serde_json::json!({
                            "type": "thinking",
                            "content": thinking,
                        }));
                    }
                }
                if let Some(content) = chunk.text() {
                    if !content.is_empty() {
                        let _ = w.try_send(serde_json::json!({
                            "type": "token",
                            "content": content,
                        }));
                    }
                }
            }
            if let Some(thinking) = chunk.thinking() {
                accumulated_thinking.push_str(thinking);
            }
            if let Some(content) = chunk.text() {
                accumulated_content.push_str(content);
            }
        }
    }

    // Build the final Message from accumulated content + tool calls (if any).
    let mut final_message = match tool_calls_message {
        Some(tc_msg) => {
            // Reconstruct with full accumulated content + the assembled tool calls.
            let tool_calls = match tc_msg {
                Message::Ai { tool_calls, .. } => tool_calls,
                _ => vec![],
            };
            Message::ai_with_tool_calls(accumulated_content, tool_calls)
        }
        None => Message::ai(accumulated_content),
    };

    if !accumulated_thinking.is_empty() {
        if let Message::Ai {
            thinking: ref mut th,
            ..
        } = final_message
        {
            *th = Some(accumulated_thinking);
        }
    }

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

/// Print the last AI message from an `invoke` / `ainvoke` result.
///
/// Mirrors [`print_stream`] for non-streaming scenarios. Finds the last
/// AI message in the `{"messages": [...]}` state, prints its thinking (if any)
/// in dim gray followed by the content in normal color.
///
/// # Example
/// ```ignore
/// let result = agent.ainvoke(&input, &RunnableConfig::new()).await?;
/// print_result(&result);
/// ```
pub fn print_result(result: &JsonValue) {
    print_result_with_options(result, true);
}

/// Like [`print_result`] but with explicit control over thinking display.
///
/// When `show_thinking` is `false` the thinking block is omitted, matching the
/// behaviour of [`print_stream_with_options`] with `show_thinking = false`.
pub fn print_result_with_options(result: &JsonValue, show_thinking: bool) {
    let messages = match result.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return,
    };

    // Walk backwards to find the last AI message that has non-empty content.
    for msg in messages.iter().rev() {
        if msg.get("type").and_then(|t| t.as_str()) != Some("ai") {
            continue;
        }

        // Print thinking in dim gray (same ANSI codes as stream_and_print).
        if show_thinking {
            if let Some(thinking) = msg.get("thinking").and_then(|t| t.as_str()) {
                if !thinking.is_empty() {
                    println!("\x1b[2;90m[Thinking] {}\x1b[0m", thinking);
                }
            }
        }

        // Print the answer content.
        if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                println!("{}", content);
                return;
            }
        }

        // Fallback: mention tool calls if the last AI turn was a tool-call step.
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            if !tool_calls.is_empty() {
                println!("[Called {} tool(s)]", tool_calls.len());
                return;
            }
        }
    }
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

/// Stream graph execution and print tokens to stdout in real-time.
///
/// Handles the common streaming boilerplate in examples. Tokens from
/// `StreamMode::Custom` are printed inline (typewriter style). Node
/// completion updates from `StreamMode::Updates` are printed as `[update]` lines.
/// Thinking content is printed in dim gray with a `[Thinking]` prefix.
///
/// Returns the collected token text.
///
/// # Example
/// ```ignore
/// use langgraph::prelude::*;
/// use langgraph_prebuilt::print_stream;
///
/// let mut stream = app.astream(&input, &RunnableConfig::new(), vec![StreamMode::Custom, StreamMode::Updates]);
/// let text = print_stream(&mut stream, true).await;
/// println!("Final: {}", text);
/// ```
pub async fn print_stream(
    stream: &mut (impl StreamExt<Item = StreamPart> + Unpin),
    print_updates: bool,
) -> String {
    print_stream_with_options(stream, print_updates, true).await
}

/// Like [`print_stream`] but with explicit control over thinking display.
///
/// When `show_thinking` is `false`, thinking/reasoning content is suppressed.
pub async fn print_stream_with_options(
    stream: &mut (impl StreamExt<Item = StreamPart> + Unpin),
    print_updates: bool,
    show_thinking: bool,
) -> String {
    let mut collected = String::new();
    let mut in_thinking = false;

    while let Some(part) = stream.next().await {
        match part.mode {
            StreamMode::Custom => {
                if let Some(token_type) = part.data.get("type").and_then(|t| t.as_str()) {
                    match token_type {
                        "thinking" if show_thinking => {
                            if let Some(content) = part.data.get("content").and_then(|c| c.as_str())
                            {
                                if !in_thinking {
                                    // ANSI dark gray: ESC[2;90m — dim + bright black
                                    // Resets at the end of each thinking block.
                                    print!("\x1b[2;90m[Thinking] ");
                                    in_thinking = true;
                                }
                                print!("{}", content);
                                let _ = std::io::stdout().flush();
                            }
                        }
                        "token" => {
                            if in_thinking {
                                // End of thinking block — reset color, then new line before answer
                                print!("\x1b[0m");
                                println!();
                                in_thinking = false;
                            }
                            if let Some(content) = part.data.get("content").and_then(|c| c.as_str())
                            {
                                print!("{}", content);
                                let _ = std::io::stdout().flush();
                                collected.push_str(content);
                            }
                        }
                        _ => {}
                    }
                }
            }
            StreamMode::Updates if print_updates => {
                if in_thinking {
                    print!("\x1b[0m");
                    println!();
                    in_thinking = false;
                }
                if let Some(obj) = part.data.as_object() {
                    for (node_name, _) in obj {
                        println!("\n[update] Node '{}' completed", node_name);
                    }
                }
            }
            _ => {}
        }
    }

    if in_thinking {
        print!("\x1b[0m");
        println!();
    }

    collected
}
