//! Type conversions between langgraph-prebuilt Message types and async-openai types.

use async_openai::types::{
    ChatCompletionMessageToolCall,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs,
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionCall,
    FunctionObjectArgs,
};
use langgraph_prebuilt::{Message, ToolCall, ToolDef};

/// Convert a langgraph-prebuilt Message to an OpenAI ChatCompletionRequestMessage.
pub fn to_openai_message(msg: &Message) -> Option<ChatCompletionRequestMessage> {
    match msg {
        Message::Human { content, .. } => {
            let text = content_text(content);
            let builder = ChatCompletionRequestUserMessageArgs::default()
                .content(text)
                .build()
                .ok()?;
            Some(ChatCompletionRequestMessage::User(builder))
        }
        Message::Ai {
            content,
            tool_calls,
            ..
        } => {
            let text = content_text(content);
            let mut builder = ChatCompletionRequestAssistantMessageArgs::default();
            if !text.is_empty() {
                builder.content(text);
            }
            if !tool_calls.is_empty() {
                let calls: Vec<ChatCompletionMessageToolCall> = tool_calls
                    .iter()
                    .enumerate()
                    .map(|(i, tc)| ChatCompletionMessageToolCall {
                        id: tc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("call_{}", i)),
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionCall {
                            name: tc.name.clone(),
                            arguments: serde_json::to_string(&tc.args).unwrap_or_default(),
                        },
                    })
                    .collect();
                builder.tool_calls(calls);
            }
            let built = builder.build().ok()?;
            Some(ChatCompletionRequestMessage::Assistant(built))
        }
        Message::System { content, .. } => {
            let text = content_text(content);
            let builder = ChatCompletionRequestSystemMessageArgs::default()
                .content(text)
                .build()
                .ok()?;
            Some(ChatCompletionRequestMessage::System(builder))
        }
        Message::Tool {
            content,
            tool_call_id,
            ..
        } => {
            let text = content_text(content);
            let builder = ChatCompletionRequestToolMessageArgs::default()
                .content(text)
                .tool_call_id(tool_call_id.as_str())
                .build()
                .ok()?;
            Some(ChatCompletionRequestMessage::Tool(builder))
        }
        Message::Remove { .. } => None, // Remove messages don't get sent to the model
    }
}

/// Convert an OpenAI tool call to a langgraph-prebuilt ToolCall.
pub fn from_openai_tool_call(tc: &ChatCompletionMessageToolCall) -> ToolCall {
    let args: serde_json::Value =
        serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
    ToolCall {
        name: tc.function.name.clone(),
        args,
        id: Some(tc.id.clone()),
    }
}

/// Convert a langgraph-prebuilt ToolDef to an OpenAI ChatCompletionTool.
pub fn to_openai_tool(def: &ToolDef) -> ChatCompletionTool {
    let mut func_builder = FunctionObjectArgs::default();
    func_builder
        .name(&def.name)
        .description(&def.description)
        .parameters(def.parameters.clone());

    ChatCompletionToolArgs::default()
        .r#type(ChatCompletionToolType::Function)
        .function(func_builder.build().unwrap_or_default())
        .build()
        .unwrap_or_default()
}

/// Extract text from MessageContent.
fn content_text(content: &langgraph_prebuilt::MessageContent) -> String {
    match content {
        langgraph_prebuilt::MessageContent::Text(s) => s.clone(),
        langgraph_prebuilt::MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                langgraph_prebuilt::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}
