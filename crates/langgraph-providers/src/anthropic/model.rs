use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::{
    BaseChatModel, LlmUsage, Message, MessageStream, ModelError, ToolCall, ToolDef,
};

use crate::common;

// ── Request types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct RawRequest {
    model: String,
    messages: Vec<RawMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<RawToolDef>>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
}

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Serialize)]
struct RawMessage {
    role: String,
    content: RawContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize, Deserialize, Clone)]
struct ImageSource {
    #[serde(rename = "type")]
    kind: String,
    url: String,
}


#[derive(Serialize)]
struct RawToolDef {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ── Response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawResponse {
    content: Vec<ResponseContentBlock>,
    #[serde(default)]
    usage: Option<RawUsage>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    Thinking { thinking: String },
}

#[derive(Deserialize)]
struct RawUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

// ── Streaming types ────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamEvent {
    MessageStart {
        message: StreamMessage,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        #[serde(default)]
        usage: Option<StreamUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
}

#[derive(Deserialize)]
struct StreamMessage {
    #[serde(default)]
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
    Thinking { thinking: String },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
}

#[derive(Deserialize)]
struct MessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamUsage {
    #[serde(default)]
    output_tokens: u32,
}

#[derive(Deserialize)]
struct StreamError {
    message: String,
}

// ── AnthropicModelConfig ──────────────────────────────────────────

/// Configuration for the Anthropic Claude model.
#[derive(Debug, Clone)]
pub struct AnthropicModelConfig {
    /// Model name (e.g., "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001").
    pub model: String,
    /// API key.
    pub api_key: String,
    /// Optional API base URL (defaults to https://api.anthropic.com).
    pub api_base: Option<String>,
    /// Optional API version header (defaults to 2023-06-01).
    pub api_version: Option<String>,
    /// Temperature for sampling (0.0 - 1.0).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate (required by Anthropic API).
    pub max_tokens: u32,
    /// Top-p sampling.
    pub top_p: Option<f32>,
    /// Top-k sampling.
    pub top_k: Option<u32>,
    /// Stop sequences.
    pub stop_sequences: Option<Vec<String>>,
}

impl Default for AnthropicModelConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            api_key: String::new(),
            api_base: None,
            api_version: None,
            temperature: None,
            max_tokens: 4096,
            top_p: None,
            top_k: None,
            stop_sequences: None,
        }
    }
}

// ── AnthropicModel ────────────────────────────────────────────────

/// Anthropic Claude model implementation using raw HTTP requests.
pub struct AnthropicModel {
    client: reqwest::Client,
    config: AnthropicModelConfig,
    tools: Vec<ToolDef>,
}

impl AnthropicModel {
    /// Create a new Anthropic model with the given configuration.
    pub fn new(config: AnthropicModelConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            tools: Vec::new(),
        }
    }

    /// Create an Anthropic model from environment variables.
    /// Reads ANTHROPIC_API_KEY and optionally ANTHROPIC_API_BASE.
    pub fn from_env() -> Result<Self, ModelError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ModelError::Config("ANTHROPIC_API_KEY not set".to_string()))?;
        let api_base = std::env::var("ANTHROPIC_API_BASE").ok();

        Ok(Self::new(AnthropicModelConfig {
            api_key,
            api_base,
            ..Default::default()
        }))
    }

    /// Create an Anthropic model from environment with a specific model name.
    pub fn from_env_with_model(model: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| ModelError::Config("ANTHROPIC_API_KEY not set".to_string()))?;
        let api_base = std::env::var("ANTHROPIC_API_BASE").ok();

        Ok(Self::new(AnthropicModelConfig {
            model: model.into(),
            api_key,
            api_base,
            ..Default::default()
        }))
    }

    fn api_url(&self) -> String {
        let base = self
            .config
            .api_base
            .as_deref()
            .unwrap_or("https://api.anthropic.com")
            .trim_end_matches('/');
        format!("{}/v1/messages", base)
    }

    fn api_version(&self) -> &str {
        self.config
            .api_version
            .as_deref()
            .unwrap_or("2023-06-01")
    }

    fn convert_content(content: &langgraph_prebuilt::MessageContent) -> RawContent {
        match content {
            langgraph_prebuilt::MessageContent::Text(s) => RawContent::Text(s.clone()),
            langgraph_prebuilt::MessageContent::Blocks(blocks) => {
                let raw_blocks: Vec<ContentBlock> = blocks
                    .iter()
                    .map(|block| match block {
                        langgraph_prebuilt::ContentBlock::Text { text } => {
                            ContentBlock::Text { text: text.clone() }
                        }
                        langgraph_prebuilt::ContentBlock::ImageUrl { image_url } => {
                            ContentBlock::Image {
                                source: ImageSource {
                                    kind: "url".to_string(),
                                    url: image_url.url.clone(),
                                },
                            }
                        }
                    })
                    .collect();
                RawContent::Blocks(raw_blocks)
            }
        }
    }

    /// Build the request, extracting system messages to top-level.
    fn build_request(
        &self,
        messages: &[Message],
        stream: bool,
    ) -> RawRequest {
        let mut system_text = Vec::new();
        let mut raw_messages = Vec::new();

        for msg in messages {
            match msg {
                Message::System { content, .. } => {
                    system_text.push(common::content_text(content));
                }
                Message::Human { content, .. } => {
                    raw_messages.push(RawMessage {
                        role: "user".to_string(),
                        content: Self::convert_content(content),
                    });
                }
                Message::Ai {
                    content,
                    tool_calls,
                    thinking,
                    ..
                } => {
                    let mut blocks = Vec::new();
                    if let Some(t) = thinking {
                        blocks.push(ContentBlock::Thinking {
                            thinking: t.clone(),
                            signature: None,
                        });
                    }
                    match content {
                        langgraph_prebuilt::MessageContent::Text(text) => {
                            if !text.is_empty() {
                                blocks.push(ContentBlock::Text { text: text.clone() });
                            }
                        }
                        langgraph_prebuilt::MessageContent::Blocks(b_list) => {
                            for b in b_list {
                                match b {
                                    langgraph_prebuilt::ContentBlock::Text { text } => {
                                        blocks.push(ContentBlock::Text { text: text.clone() });
                                    }
                                    langgraph_prebuilt::ContentBlock::ImageUrl { image_url } => {
                                        blocks.push(ContentBlock::Image {
                                            source: ImageSource {
                                                kind: "url".to_string(),
                                                url: image_url.url.clone(),
                                            },
                                        });
                                    }
                                }
                            }
                        }
                    }
                    for tc in tool_calls {
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id.clone().unwrap_or_else(|| "toolu_0".to_string()),
                            name: tc.name.clone(),
                            input: tc.args.clone(),
                        });
                    }
                    if blocks.is_empty() {
                        blocks.push(ContentBlock::Text {
                            text: String::new(),
                        });
                    }
                    raw_messages.push(RawMessage {
                        role: "assistant".to_string(),
                        content: RawContent::Blocks(blocks),
                    });
                }
                Message::Tool {
                    content,
                    tool_call_id,
                    ..
                } => {
                    let mut blocks = Vec::new();
                    match content {
                        langgraph_prebuilt::MessageContent::Text(text) => {
                            blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call_id.clone(),
                                content: text.clone(),
                            });
                        }
                        langgraph_prebuilt::MessageContent::Blocks(b_list) => {
                            // Convert back to text since ToolResult content in Anthropic requires a string,
                            // or represent it appropriately. For simple tool content in Anthropic, we can use the text conversion.
                            let text = common::content_text(content);
                            blocks.push(ContentBlock::ToolResult {
                                tool_use_id: tool_call_id.clone(),
                                content: text,
                            });
                        }
                    }
                    raw_messages.push(RawMessage {
                        role: "user".to_string(),
                        content: RawContent::Blocks(blocks),
                    });
                }
                Message::Remove { .. } => {}
            }
        }

        let system = if system_text.is_empty() {
            None
        } else {
            Some(system_text.join("\n"))
        };

        let tools = if self.tools.is_empty() {
            None
        } else {
            Some(
                self.tools
                    .iter()
                    .map(|t| RawToolDef {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: t.parameters.clone(),
                    })
                    .collect(),
            )
        };

        RawRequest {
            model: self.config.model.clone(),
            messages: raw_messages,
            system,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            stop_sequences: self.config.stop_sequences.clone(),
            tools,
            stream,
        }
    }

    fn extract_usage(raw: &RawUsage) -> LlmUsage {
        LlmUsage {
            prompt_tokens: raw.input_tokens,
            completion_tokens: raw.output_tokens,
            total_tokens: raw.input_tokens + raw.output_tokens,
        }
    }

    fn extract_tool_calls(blocks: &[ResponseContentBlock]) -> Vec<ToolCall> {
        blocks
            .iter()
            .filter_map(|b| match b {
                ResponseContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                    name: name.clone(),
                    args: input.clone(),
                    id: Some(id.clone()),
                }),
                _ => None,
            })
            .collect()
    }
}

#[async_trait]
impl BaseChatModel for AnthropicModel {
    fn name(&self) -> &str {
        "Anthropic"
    }

    fn invoke(
        &self,
        messages: &[Message],
        _config: &RunnableConfig,
    ) -> Result<Message, ModelError> {
        common::invoke_sync(self.ainvoke(messages, _config))
    }

    async fn ainvoke(
        &self,
        messages: &[Message],
        _config: &RunnableConfig,
    ) -> Result<Message, ModelError> {
        let request = self.build_request(messages, false);

        let response = self
            .client
            .post(self.api_url())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", self.api_version())
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| ModelError::Invocation(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ModelError::Invocation(format!(
                "API error {}: {}",
                status, body
            )));
        }

        let raw: RawResponse = response
            .json()
            .await
            .map_err(|e| ModelError::Invocation(e.to_string()))?;

        // Extract text content
        let content: String = raw
            .content
            .iter()
            .filter_map(|b| match b {
                ResponseContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        // Extract thinking
        let thinking: Option<String> = raw.content.iter().find_map(|b| match b {
            ResponseContentBlock::Thinking { thinking } => Some(thinking.clone()),
            _ => None,
        });

        let tool_calls = Self::extract_tool_calls(&raw.content);
        let usage = raw.usage.as_ref().map(Self::extract_usage);

        Ok(common::build_ai_message(content, tool_calls, thinking, usage))
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        _config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        Box::pin(async_stream::stream! {
            let request = self.build_request(messages, true);

            let response = self
                .client
                .post(self.api_url())
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", self.api_version())
                .header("content-type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| ModelError::Invocation(e.to_string()))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                yield Err(ModelError::Invocation(format!("API error {}: {}", status, body)));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            let mut accumulated_content = String::new();
            let mut accumulated_thinking = String::new();
            let mut tool_call_buffers: Vec<(Option<String>, String, String)> = Vec::new();
            let mut usage: Option<LlmUsage> = None;
            let mut current_block_index: Option<usize> = None;

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(|e| ModelError::Invocation(e.to_string()))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events from buffer
                while let Some(event_end) = find_sse_event_end(&buffer) {
                    let event_str = buffer[..event_end].to_string();
                    buffer = buffer[event_end..].to_string();

                    // Parse SSE: "event: <type>\ndata: <json>\n\n"
                    let mut event_type = "";
                    let mut data_line = "";
                    for line in event_str.lines() {
                        if let Some(rest) = line.strip_prefix("event: ") {
                            event_type = rest.trim();
                        } else if let Some(rest) = line.strip_prefix("data: ") {
                            data_line = rest.trim();
                        }
                    }

                    if data_line.is_empty() {
                        continue;
                    }

                    // Handle based on event type
                    match event_type {
                        "message_start" => {
                            if let Ok(ev) = serde_json::from_str::<MessageStartEvent>(data_line) {
                                if let Some(u) = ev.message.usage {
                                    usage = Some(Self::extract_usage(&u));
                                }
                            }
                        }
                        "content_block_start" => {
                            if let Ok(ev) = serde_json::from_str::<ContentBlockStartEvent>(data_line) {
                                current_block_index = Some(ev.index);
                                match ev.content_block {
                                    ContentBlockStart::ToolUse { id, name } => {
                                        while tool_call_buffers.len() <= ev.index {
                                            tool_call_buffers.push((None, String::new(), String::new()));
                                        }
                                        tool_call_buffers[ev.index].0 = Some(id);
                                        tool_call_buffers[ev.index].1 = name;
                                    }
                                    ContentBlockStart::Thinking { thinking } => {
                                        accumulated_thinking.push_str(&thinking);
                                        yield Ok(Message::ai_with_thinking("", thinking));
                                    }
                                    _ => {}
                                }
                            }
                        }
                        "content_block_delta" => {
                            if let Ok(ev) = serde_json::from_str::<ContentBlockDeltaEvent>(data_line) {
                                match ev.delta {
                                    ContentBlockDelta::TextDelta { text } => {
                                        accumulated_content.push_str(&text);
                                        yield Ok(Message::ai(text));
                                    }
                                    ContentBlockDelta::ThinkingDelta { thinking } => {
                                        accumulated_thinking.push_str(&thinking);
                                        yield Ok(Message::ai_with_thinking("", thinking));
                                    }
                                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                                        if let Some(idx) = current_block_index {
                                            if idx < tool_call_buffers.len() {
                                                tool_call_buffers[idx].2.push_str(&partial_json);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "content_block_stop" => {
                            current_block_index = None;
                        }
                        "message_delta" => {
                            if let Ok(ev) = serde_json::from_str::<MessageDeltaEvent>(data_line) {
                                if let Some(u) = ev.usage {
                                    let prompt = usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
                                    usage = Some(LlmUsage {
                                        prompt_tokens: prompt,
                                        completion_tokens: u.output_tokens,
                                        total_tokens: prompt + u.output_tokens,
                                    });
                                }
                            }
                        }
                        "message_stop" => {
                            break;
                        }
                        "ping" | _ => {}
                    }
                }
            }

            // Yield final assembled tool calls
            if !tool_call_buffers.is_empty() {
                let tool_calls: Vec<ToolCall> = tool_call_buffers
                    .into_iter()
                    .filter(|(_, name, _)| !name.is_empty())
                    .map(|(id, name, args)| {
                        let args_json = if args.is_empty() {
                            serde_json::json!({})
                        } else {
                            serde_json::from_str(&args).unwrap_or(serde_json::json!({}))
                        };
                        ToolCall { name, args: args_json, id }
                    })
                    .collect();

                let thinking = if accumulated_thinking.is_empty() { None } else { Some(accumulated_thinking) };
                yield Ok(common::build_ai_message(String::new(), tool_calls, thinking, usage));
            } else if usage.is_some() {
                let thinking = if accumulated_thinking.is_empty() { None } else { Some(accumulated_thinking) };
                yield Ok(common::build_ai_message(String::new(), Vec::new(), thinking, usage));
            }
        })
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        let mut new_model = AnthropicModel::new(self.config.clone());
        new_model.tools = tools;
        Box::new(new_model)
    }
}

// ── Helper: find end of an SSE event block ─────────────────────────

fn find_sse_event_end(buf: &str) -> Option<usize> {
    // SSE events are terminated by \n\n or \r\n\r\n
    if let Some(pos) = buf.find("\n\n") {
        return Some(pos + 2);
    }
    if let Some(pos) = buf.find("\r\n\r\n") {
        return Some(pos + 4);
    }
    None
}

// ── Minimal wrappers for serde event parsing ───────────────────────

#[derive(Deserialize)]
struct MessageStartEvent {
    message: StreamMessage,
}

#[derive(Deserialize)]
struct ContentBlockStartEvent {
    index: usize,
    content_block: ContentBlockStart,
}

#[derive(Deserialize)]
struct ContentBlockDeltaEvent {
    delta: ContentBlockDelta,
}

#[derive(Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDelta,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AnthropicModelConfig::default();
        assert_eq!(config.model, "claude-sonnet-4-20250514");
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_build_request() {
        let model = AnthropicModel::new(AnthropicModelConfig {
            model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test-key".to_string(),
            ..Default::default()
        });

        let messages = vec![
            Message::system("You are helpful."),
            Message::human("Hello"),
        ];
        let req = model.build_request(&messages, false);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.system.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn test_bind_tools() {
        let model = AnthropicModel::new(AnthropicModelConfig {
            model: "claude-sonnet-4-20250514".to_string(),
            api_key: "test-key".to_string(),
            ..Default::default()
        });

        let tool = ToolDef {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        };

        let bound = model.bind_tools(vec![tool]);
        assert_eq!(bound.name(), "Anthropic");
    }

    #[test]
    fn test_extract_tool_calls() {
        let blocks = vec![
            ResponseContentBlock::Text {
                text: "Let me search".to_string(),
            },
            ResponseContentBlock::ToolUse {
                id: "toolu_123".to_string(),
                name: "search".to_string(),
                input: serde_json::json!({"query": "rust"}),
            },
        ];
        let calls = AnthropicModel::extract_tool_calls(&blocks);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].id.as_deref(), Some("toolu_123"));
    }
}
