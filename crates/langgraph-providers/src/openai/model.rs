use async_trait::async_trait;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};

use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::{
    BaseChatModel, LlmUsage, Message, MessageStream, ModelError, ToolCall, ToolDef,
};

use crate::common;

// ── Request types ──────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(untagged)]
enum RawContent {
    Text(String),
    Blocks(Vec<OpenAIContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIContentBlock {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: OpenAIImageUrl,
    },
}

#[derive(Serialize)]
struct OpenAIImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Serialize)]
struct RawMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<RawContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<RawToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}


#[derive(Serialize, Clone)]
struct RawToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: RawFunctionCall,
}

#[derive(Serialize, Clone)]
struct RawFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct RawToolDef {
    #[serde(rename = "type")]
    kind: String,
    function: RawFunctionObject,
}

#[derive(Serialize)]
struct RawFunctionObject {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct RawRequest {
    model: String,
    messages: Vec<RawMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<RawToolDef>>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

// ── Response types ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawResponse {
    choices: Vec<RawChoice>,
    #[serde(default)]
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct RawChoice {
    message: RawResponseMessage,
}

#[derive(Deserialize)]
struct RawResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RawToolCallResp>>,
    /// Thinking/reasoning content from reasoning models.
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct RawToolCallResp {
    id: String,
    function: RawFunctionCallResp,
}

#[derive(Deserialize)]
struct RawFunctionCallResp {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct RawUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

// ── Streaming types ────────────────────────────────────────────────

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<RawUsage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunction>,
}

#[derive(Deserialize)]
struct StreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ── OpenAIModelConfig ──────────────────────────────────────────────

/// Configuration for the OpenAI model.
#[derive(Debug, Clone)]
pub struct OpenAIModelConfig {
    /// Model name (e.g., "gpt-4o", "o1", "deepseek-reasoner").
    pub model: String,
    /// API key.
    pub api_key: String,
    /// Optional API base URL (defaults to https://api.openai.com/v1).
    pub api_base: Option<String>,
    /// Temperature for sampling (0.0 - 2.0).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Top-p sampling.
    pub top_p: Option<f32>,
    /// Frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    pub presence_penalty: Option<f32>,
}

impl Default for OpenAIModelConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            api_key: String::new(),
            api_base: None,
            temperature: None,
            max_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

// ── OpenAIModel ────────────────────────────────────────────────────

/// OpenAI chat model implementation using raw HTTP requests.
///
/// Supports all OpenAI-compatible APIs including reasoning models that return
/// `reasoning_content` (DeepSeek, SiliconFlow, etc.).
pub struct OpenAIModel {
    client: reqwest::Client,
    config: OpenAIModelConfig,
    tools: Vec<ToolDef>,
}

impl OpenAIModel {
    /// Create a new OpenAI model with the given configuration.
    pub fn new(config: OpenAIModelConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            tools: Vec::new(),
        }
    }

    /// Create an OpenAI model from environment variables.
    /// Reads OPENAI_API_KEY and optionally OPENAI_API_BASE.
    pub fn from_env() -> Result<Self, ModelError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ModelError::Config("OPENAI_API_KEY not set".to_string()))?;
        let api_base = std::env::var("OPENAI_API_BASE").ok();

        Ok(Self::new(OpenAIModelConfig {
            api_key,
            api_base,
            ..Default::default()
        }))
    }

    /// Create an OpenAI model from environment with a specific model name.
    pub fn from_env_with_model(model: impl Into<String>) -> Result<Self, ModelError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| ModelError::Config("OPENAI_API_KEY not set".to_string()))?;
        let api_base = std::env::var("OPENAI_API_BASE").ok();

        Ok(Self::new(OpenAIModelConfig {
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
            .unwrap_or("https://api.openai.com/v1")
            .trim_end_matches('/');
        format!("{}/chat/completions", base)
    }

    fn convert_content(content: &langgraph_prebuilt::MessageContent) -> Option<RawContent> {
        match content {
            langgraph_prebuilt::MessageContent::Text(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(RawContent::Text(s.clone()))
                }
            }
            langgraph_prebuilt::MessageContent::Blocks(blocks) => {
                let raw_blocks: Vec<OpenAIContentBlock> = blocks
                    .iter()
                    .map(|block| match block {
                        langgraph_prebuilt::ContentBlock::Text { text } => {
                            OpenAIContentBlock::Text { text: text.clone() }
                        }
                        langgraph_prebuilt::ContentBlock::ImageUrl { image_url } => {
                            OpenAIContentBlock::ImageUrl {
                                image_url: OpenAIImageUrl {
                                    url: image_url.url.clone(),
                                    detail: image_url.detail.clone(),
                                },
                            }
                        }
                    })
                    .collect();
                if raw_blocks.is_empty() {
                    None
                } else {
                    Some(RawContent::Blocks(raw_blocks))
                }
            }
        }
    }

    fn build_messages(&self, messages: &[Message]) -> Vec<RawMessage> {
        messages
            .iter()
            .filter_map(|msg| match msg {
                Message::Human { content, .. } => Some(RawMessage {
                    role: "user".to_string(),
                    content: Self::convert_content(content),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }),
                Message::Ai {
                    content,
                    tool_calls,
                    thinking,
                    ..
                } => {
                    let tc = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(
                            tool_calls
                                .iter()
                                .enumerate()
                                .map(|(i, tc)| RawToolCall {
                                    id: tc
                                        .id
                                        .clone()
                                        .unwrap_or_else(|| format!("call_{}", i)),
                                    kind: "function".to_string(),
                                    function: RawFunctionCall {
                                        name: tc.name.clone(),
                                        arguments: serde_json::to_string(&tc.args)
                                            .unwrap_or_default(),
                                    },
                                })
                                .collect(),
                        )
                    };
                    Some(RawMessage {
                        role: "assistant".to_string(),
                        content: Self::convert_content(content),
                        tool_calls: tc,
                        tool_call_id: None,
                        reasoning_content: thinking.clone(),
                    })
                }
                Message::System { content, .. } => Some(RawMessage {
                    role: "system".to_string(),
                    content: Self::convert_content(content),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }),
                Message::Tool {
                    content,
                    tool_call_id,
                    ..
                } => Some(RawMessage {
                    role: "tool".to_string(),
                    content: Self::convert_content(content),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.clone()),
                    reasoning_content: None,
                }),
                Message::Remove { .. } => None,
            })
            .collect()
    }

    fn build_tools(&self) -> Option<Vec<RawToolDef>> {
        if self.tools.is_empty() {
            return None;
        }
        Some(
            self.tools
                .iter()
                .map(|t| RawToolDef {
                    kind: "function".to_string(),
                    function: RawFunctionObject {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        )
    }

    fn extract_usage(raw: &RawUsage) -> LlmUsage {
        LlmUsage {
            prompt_tokens: raw.prompt_tokens,
            completion_tokens: raw.completion_tokens,
            total_tokens: raw.total_tokens,
        }
    }

}

#[async_trait]
impl BaseChatModel for OpenAIModel {
    fn name(&self) -> &str {
        "OpenAI"
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
        let request = RawRequest {
            model: self.config.model.clone(),
            messages: self.build_messages(messages),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            top_p: self.config.top_p,
            frequency_penalty: self.config.frequency_penalty,
            presence_penalty: self.config.presence_penalty,
            tools: self.build_tools(),
            stream: false,
            stream_options: None,
        };

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
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

        let choice = raw
            .choices
            .first()
            .ok_or_else(|| ModelError::Invocation("No choices in response".to_string()))?;

        let content = choice.message.content.clone().unwrap_or_default();
        let thinking = choice.message.reasoning_content.clone();
        let usage = raw.usage.as_ref().map(Self::extract_usage);

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|tc| {
                        let args = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        ToolCall {
                            name: tc.function.name.clone(),
                            args,
                            id: Some(tc.id.clone()),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(common::build_ai_message(content, tool_calls, thinking, usage))
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        _config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        Box::pin(async_stream::stream! {
            let request = RawRequest {
                model: self.config.model.clone(),
                messages: self.build_messages(messages),
                temperature: self.config.temperature,
                max_tokens: self.config.max_tokens,
                top_p: self.config.top_p,
                frequency_penalty: self.config.frequency_penalty,
                presence_penalty: self.config.presence_penalty,
                tools: self.build_tools(),
                stream: true,
                stream_options: Some(StreamOptions { include_usage: true }),
            };

            let es_builder = self
                .client
                .post(self.api_url())
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .header("Content-Type", "application/json")
                .json(&request);

            let mut event_source = es_builder
                .eventsource()
                .map_err(|e| ModelError::Invocation(e.to_string()))?;

            let mut accumulated_content = String::new();
            let mut accumulated_thinking = String::new();
            let mut tool_call_buffers: Vec<(Option<String>, String, String)> = Vec::new();
            let mut usage: Option<LlmUsage> = None;

            while let Some(event) = event_source.next().await {
                let event = event.map_err(|e| ModelError::Invocation(e.to_string()))?;

                match event {
                    Event::Open => continue,
                    Event::Message(msg) => {
                        if msg.data == "[DONE]" {
                            break;
                        }

                        let chunk: StreamChunk = serde_json::from_str(&msg.data)
                            .map_err(|e| ModelError::Invocation(e.to_string()))?;

                        if let Some(u) = chunk.usage {
                            usage = Some(Self::extract_usage(&u));
                        }

                        if let Some(choice) = chunk.choices.first() {
                            let delta = &choice.delta;

                            // Stream thinking delta — incremental only
                            if let Some(ref thinking) = delta.reasoning_content {
                                accumulated_thinking.push_str(thinking);
                                yield Ok(Message::ai_with_thinking("", thinking.clone()));
                            }

                            // Stream answer delta — incremental only
                            if let Some(ref content) = delta.content {
                                accumulated_content.push_str(content);
                                yield Ok(Message::ai(content.clone()));
                            }

                            // Accumulate tool call fragments (not yielded until done)
                            if let Some(calls) = &delta.tool_calls {
                                for tc in calls {
                                    let idx = tc.index;
                                    while tool_call_buffers.len() <= idx {
                                        tool_call_buffers.push((None, String::new(), String::new()));
                                    }
                                    let buf = &mut tool_call_buffers[idx];
                                    if let Some(id) = &tc.id {
                                        buf.0 = Some(id.clone());
                                    }
                                    if let Some(func) = &tc.function {
                                        if let Some(name) = &func.name {
                                            buf.1 = name.clone();
                                        }
                                        if let Some(args) = &func.arguments {
                                            buf.2.push_str(args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // After the SSE stream ends, yield ONE final chunk containing only
            // the assembled tool calls (if any). Content/thinking are left empty
            // because they have already been streamed incrementally above.
            // Consumers that only need the final assembled Message (e.g. invoke)
            // should call `ainvoke` instead. Consumers of `astream` that need
            // tool calls can detect this chunk via `has_tool_calls()`.
            if !tool_call_buffers.is_empty() {
                let tool_calls: Vec<ToolCall> = tool_call_buffers
                    .into_iter()
                    .filter(|(_, name, _)| !name.is_empty())
                    .map(|(id, name, args)| {
                        let args_json = serde_json::from_str(&args)
                            .unwrap_or(serde_json::json!({}));
                        ToolCall { name, args: args_json, id }
                    })
                    .collect();

                yield Ok(common::build_ai_message(String::new(), tool_calls, None, usage));
            } else if usage.is_some() {
                yield Ok(common::build_ai_message(String::new(), Vec::new(), None, usage));
            }
        })
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        let mut new_model = OpenAIModel::new(self.config.clone());
        new_model.tools = tools;
        Box::new(new_model)
    }
}

// ── OpenAICompatModel ──────────────────────────────────────────────

/// An OpenAI-compatible model for use with alternative endpoints
/// (e.g., DeepSeek, Ollama, vLLM, LiteLLM, Azure OpenAI).
pub struct OpenAICompatModel {
    inner: OpenAIModel,
}

impl OpenAICompatModel {
    /// Create a new compatible model with a custom base URL.
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            inner: OpenAIModel::new(OpenAIModelConfig {
                model: model.into(),
                api_key: api_key.into(),
                api_base: Some(base_url.into()),
                ..Default::default()
            }),
        }
    }
}

#[async_trait]
impl BaseChatModel for OpenAICompatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn invoke(&self, messages: &[Message], config: &RunnableConfig) -> Result<Message, ModelError> {
        self.inner.invoke(messages, config)
    }

    async fn ainvoke(
        &self,
        messages: &[Message],
        config: &RunnableConfig,
    ) -> Result<Message, ModelError> {
        self.inner.ainvoke(messages, config).await
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        self.inner.astream(messages, config)
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        self.inner.bind_tools(tools)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = OpenAIModelConfig::default();
        assert_eq!(config.model, "gpt-4o");
    }

    #[test]
    fn test_build_messages() {
        let model = OpenAIModel::new(OpenAIModelConfig {
            model: "gpt-4o".to_string(),
            api_key: "test-key".to_string(),
            ..Default::default()
        });

        let messages = vec![Message::human("Hello")];
        let raw = model.build_messages(&messages);
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].role, "user");
    }

    #[test]
    fn test_bind_tools() {
        let model = OpenAIModel::new(OpenAIModelConfig {
            model: "gpt-4o".to_string(),
            api_key: "test-key".to_string(),
            ..Default::default()
        });

        let tool = ToolDef {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }),
        };

        let bound = model.bind_tools(vec![tool]);
        assert_eq!(bound.name(), "OpenAI");
    }

    #[test]
    fn test_thinking_field() {
        let msg = Message::ai_with_thinking("The answer is 4", "Let me think: 2+2=4");
        assert_eq!(msg.thinking(), Some("Let me think: 2+2=4"));
        assert_eq!(msg.text(), Some("The answer is 4"));
    }
}
