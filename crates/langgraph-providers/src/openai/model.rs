use async_openai::config::OpenAIConfig;
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs};
use async_openai::Client;
use async_trait::async_trait;
use futures::StreamExt;

use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::{BaseChatModel, Message, MessageStream, ModelError, ToolDef};

use super::types::{from_openai_tool_call, to_openai_message, to_openai_tool};

/// Configuration for the OpenAI model.
#[derive(Debug, Clone)]
pub struct OpenAIModelConfig {
    /// Model name (e.g., "gpt-4o", "gpt-4", "o1").
    pub model: String,
    /// OpenAI API key.
    pub api_key: String,
    /// Optional API base URL (for proxies or compatible APIs).
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

/// OpenAI chat model implementation.
///
/// Wraps the `async-openai` client and implements `BaseChatModel`
/// for use with LangGraph agents.
pub struct OpenAIModel {
    client: Client<OpenAIConfig>,
    config: OpenAIModelConfig,
    tools: Vec<ToolDef>,
}

impl OpenAIModel {
    /// Create a new OpenAI model with the given configuration.
    pub fn new(config: OpenAIModelConfig) -> Self {
        let mut oai_config = OpenAIConfig::new().with_api_key(&config.api_key);

        if let Some(ref base) = config.api_base {
            oai_config = oai_config.with_api_base(base);
        }

        let client = Client::with_config(oai_config);

        Self {
            client,
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

    /// Build the chat completion request.
    fn build_request(
        &self,
        messages: &[Message],
    ) -> Result<CreateChatCompletionRequestArgs, ModelError> {
        let openai_msgs: Vec<ChatCompletionRequestMessage> =
            messages.iter().filter_map(to_openai_message).collect();

        let mut builder = CreateChatCompletionRequestArgs::default();
        builder.model(&self.config.model).messages(openai_msgs);

        if let Some(temp) = self.config.temperature {
            builder.temperature(temp);
        }
        if let Some(max) = self.config.max_tokens {
            builder.max_tokens(max);
        }
        if let Some(tp) = self.config.top_p {
            builder.top_p(tp);
        }
        if let Some(fp) = self.config.frequency_penalty {
            builder.frequency_penalty(fp);
        }
        if let Some(pp) = self.config.presence_penalty {
            builder.presence_penalty(pp);
        }

        // Add tools if bound
        if !self.tools.is_empty() {
            let tool_defs: Vec<_> = self.tools.iter().map(to_openai_tool).collect();
            builder.tools(tool_defs);
        }

        Ok(builder)
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
        // Use tokio runtime for sync invocation
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // Already in a tokio context — use block_in_place
                // which is safe from within an async runtime
                tokio::task::block_in_place(|| handle.block_on(self.ainvoke(messages, _config)))
            }
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| ModelError::Invocation(e.to_string()))?;
                rt.block_on(self.ainvoke(messages, _config))
            }
        }
    }

    async fn ainvoke(
        &self,
        messages: &[Message],
        _config: &RunnableConfig,
    ) -> Result<Message, ModelError> {
        let request = self
            .build_request(messages)?
            .build()
            .map_err(|e| ModelError::Invocation(e.to_string()))?;

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .map_err(|e| ModelError::Invocation(e.to_string()))?;

        // Extract the first choice
        let choice = response
            .choices
            .first()
            .ok_or_else(|| ModelError::Invocation("No choices in response".to_string()))?;

        let msg = &choice.message;

        // Build tool calls if present
        let tool_calls: Vec<langgraph_prebuilt::ToolCall> = msg
            .tool_calls
            .as_ref()
            .map(|calls| calls.iter().map(from_openai_tool_call).collect())
            .unwrap_or_default();

        let content = msg.content.clone().unwrap_or_default();

        if tool_calls.is_empty() {
            Ok(Message::ai(content))
        } else {
            Ok(Message::ai_with_tool_calls(content, tool_calls))
        }
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        _config: &'a RunnableConfig,
    ) -> MessageStream<'a> {
        Box::pin(async_stream::stream! {
            let request = self
                .build_request(messages)?
                .stream(true)
                .build()
                .map_err(|e| ModelError::Invocation(e.to_string()))?;

            let mut stream = self
                .client
                .chat()
                .create_stream(request)
                .await
                .map_err(|e| ModelError::Invocation(e.to_string()))?;

            let mut accumulated_content = String::new();
            // Accumulate tool calls by index: (id, name, arguments_buffer)
            let mut tool_call_buffers: Vec<(Option<String>, String, String)> = Vec::new();

            while let Some(result) = stream.next().await {
                let chunk = result.map_err(|e| ModelError::Invocation(e.to_string()))?;

                if let Some(choice) = chunk.choices.first() {
                    let delta = &choice.delta;

                    // Yield delta content only (not accumulated)
                    if let Some(content) = &delta.content {
                        accumulated_content.push_str(content);
                        yield Ok(Message::ai(content.clone()));
                    }

                    // Accumulate tool calls
                    if let Some(calls) = &delta.tool_calls {
                        for tc in calls {
                            let idx = tc.index as usize;
                            // Expand buffer if needed
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

                    // Check for finish reason
                    if choice.finish_reason.is_some() {
                        break;
                    }
                }
            }

            // Build final tool calls
            if !tool_call_buffers.is_empty() {
                let tool_calls: Vec<langgraph_prebuilt::ToolCall> = tool_call_buffers
                    .into_iter()
                    .filter(|(_, name, _)| !name.is_empty())
                    .map(|(id, name, args)| {
                        let args_json = serde_json::from_str(&args)
                            .unwrap_or(serde_json::json!({}));
                        langgraph_prebuilt::ToolCall {
                            name,
                            args: args_json,
                            id,
                        }
                    })
                    .collect();

                if !tool_calls.is_empty() {
                    yield Ok(Message::ai_with_tool_calls(
                        accumulated_content,
                        tool_calls,
                    ));
                }
            }
        })
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        let mut new_model = OpenAIModel::new(self.config.clone());
        new_model.tools = tools;
        Box::new(new_model)
    }
}

/// An OpenAI-compatible model for use with local or alternative endpoints
/// (e.g., Ollama, vLLM, LiteLLM, Azure OpenAI).
///
/// Uses the same OpenAI API format but with a custom base URL.
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
    fn test_build_request() {
        let model = OpenAIModel::new(OpenAIModelConfig {
            model: "gpt-4o".to_string(),
            api_key: "test-key".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        });

        let messages = vec![Message::human("Hello")];
        let request = model.build_request(&messages);
        assert!(request.is_ok());
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
}
