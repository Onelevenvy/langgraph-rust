use langgraph_prebuilt::{ContentBlock, LlmUsage, Message, MessageContent, ModelError, ToolCall};

/// Extract plain text from MessageContent.
pub fn content_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Build a final AI Message from parts — provider-agnostic.
pub fn build_ai_message(
    content: String,
    tool_calls: Vec<ToolCall>,
    thinking: Option<String>,
    usage: Option<LlmUsage>,
) -> Message {
    let mut msg = match (tool_calls.is_empty(), usage) {
        (true, None) => Message::ai(content),
        (true, Some(u)) => Message::ai_with_usage(content, u),
        (false, None) => Message::ai_with_tool_calls(content, tool_calls),
        (false, Some(u)) => Message::ai_with_tool_calls_and_usage(content, tool_calls, u),
    };
    if let Some(t) = thinking {
        if let Message::Ai {
            thinking: ref mut th,
            ..
        } = msg
        {
            *th = Some(t);
        }
    }
    msg
}

/// Synchronous bridge to an async method — used by all providers for `invoke()`.
pub fn invoke_sync<F, T>(fut: F) -> Result<T, ModelError>
where
    F: std::future::Future<Output = Result<T, ModelError>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| ModelError::Invocation(e.to_string()))?;
            rt.block_on(fut)
        }
    }
}
