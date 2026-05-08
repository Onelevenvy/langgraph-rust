use crate::event_bus::EventBus;
use crate::store::TracingStore;
use crate::types::*;
use async_trait::async_trait;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::traits::{BaseChatModel, BaseTool, ToolDef};
use langgraph_prebuilt::types::Message;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

/// Wrapper around any BaseChatModel that records LLM call traces.
pub struct TracingChatModel<M: BaseChatModel> {
    inner: M,
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
    trace_id: String,
    parent_span_id: Option<String>,
}

impl<M: BaseChatModel> TracingChatModel<M> {
    pub fn new(
        inner: M,
        store: Arc<dyn TracingStore>,
        event_bus: EventBus,
        trace_id: String,
    ) -> Self {
        Self {
            inner,
            store,
            event_bus,
            trace_id,
            parent_span_id: None,
        }
    }

    pub fn with_parent_span(mut self, span_id: String) -> Self {
        self.parent_span_id = Some(span_id);
        self
    }
}

fn record_llm_span(
    store: &dyn TracingStore,
    event_bus: &EventBus,
    trace_id: &str,
    parent_span_id: &Option<String>,
    model_name: &str,
    input_json: JsonValue,
    result: &Result<Message, langgraph_prebuilt::traits::ModelError>,
) {
    let span_id = Uuid::new_v4().to_string();
    match result {
        Ok(response) => {
            let output_json = serde_json::to_value(response).unwrap_or(JsonValue::Null);
            let usage = response.usage();
            let mut span = Span::new(
                span_id,
                trace_id.to_string(),
                parent_span_id.clone(),
                model_name.to_string(),
                SpanType::LlmGeneration,
                input_json,
            );
            span.finish(output_json, SpanStatus::Success);
            span.metadata.model = Some(model_name.to_string());
            span.metadata.tokens_in = usage.map(|u| u.prompt_tokens);
            span.metadata.tokens_out = usage.map(|u| u.completion_tokens);
            span.metadata.total_tokens = usage.map(|u| u.total_tokens);
            store.add_span(span.clone());
            event_bus.publish(crate::event_bus::TracingEvent::SpanCreated { span });
        }
        Err(e) => {
            let mut span = Span::new(
                span_id,
                trace_id.to_string(),
                parent_span_id.clone(),
                model_name.to_string(),
                SpanType::LlmGeneration,
                input_json,
            );
            span.finish(
                serde_json::json!({"error": e.to_string()}),
                SpanStatus::Error,
            );
            span.metadata.model = Some(model_name.to_string());
            store.add_span(span.clone());
            event_bus.publish(crate::event_bus::TracingEvent::SpanCreated { span });
        }
    }
}

#[async_trait]
impl<M: BaseChatModel + 'static> BaseChatModel for TracingChatModel<M> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn invoke(
        &self,
        messages: &[Message],
        config: &RunnableConfig,
    ) -> Result<Message, langgraph_prebuilt::traits::ModelError> {
        let start = Instant::now();
        let result = self.inner.invoke(messages, config);
        let input_json = serde_json::to_value(messages).unwrap_or(JsonValue::Null);
        record_llm_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), input_json, &result);
        let _ = start;
        result
    }

    async fn ainvoke(
        &self,
        messages: &[Message],
        config: &RunnableConfig,
    ) -> Result<Message, langgraph_prebuilt::traits::ModelError> {
        let start = Instant::now();
        let result = self.inner.ainvoke(messages, config).await;
        let input_json = serde_json::to_value(messages).unwrap_or(JsonValue::Null);
        record_llm_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), input_json, &result);
        let _ = start;
        result
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        config: &'a RunnableConfig,
    ) -> langgraph_prebuilt::MessageStream<'a> {
        self.inner.astream(messages, config)
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        // We can't wrap Box<dyn BaseChatModel> in TracingChatModel because
        // Box<dyn BaseChatModel> doesn't implement BaseChatModel.
        // Instead, bind tools on the inner model and wrap the result.
        // We need to create a dynamic wrapper.
        let inner = self.inner.bind_tools(tools);
        Box::new(DynamicTracingChatModel {
            inner,
            store: self.store.clone(),
            event_bus: self.event_bus.clone(),
            trace_id: self.trace_id.clone(),
            parent_span_id: self.parent_span_id.clone(),
        })
    }
}

/// Dynamic wrapper that holds a Box<dyn BaseChatModel> instead of a generic type.
/// This is needed for bind_tools which returns Box<dyn BaseChatModel>.
struct DynamicTracingChatModel {
    inner: Box<dyn BaseChatModel>,
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
    trace_id: String,
    parent_span_id: Option<String>,
}

#[async_trait]
impl BaseChatModel for DynamicTracingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn invoke(
        &self,
        messages: &[Message],
        config: &RunnableConfig,
    ) -> Result<Message, langgraph_prebuilt::traits::ModelError> {
        let start = Instant::now();
        let result = self.inner.invoke(messages, config);
        let input_json = serde_json::to_value(messages).unwrap_or(JsonValue::Null);
        record_llm_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), input_json, &result);
        let _ = start;
        result
    }

    async fn ainvoke(
        &self,
        messages: &[Message],
        config: &RunnableConfig,
    ) -> Result<Message, langgraph_prebuilt::traits::ModelError> {
        let start = Instant::now();
        let result = self.inner.ainvoke(messages, config).await;
        let input_json = serde_json::to_value(messages).unwrap_or(JsonValue::Null);
        record_llm_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), input_json, &result);
        let _ = start;
        result
    }

    fn astream<'a>(
        &'a self,
        messages: &'a [Message],
        config: &'a RunnableConfig,
    ) -> langgraph_prebuilt::MessageStream<'a> {
        self.inner.astream(messages, config)
    }

    fn bind_tools(&self, tools: Vec<ToolDef>) -> Box<dyn BaseChatModel> {
        let inner = self.inner.bind_tools(tools);
        Box::new(DynamicTracingChatModel {
            inner,
            store: self.store.clone(),
            event_bus: self.event_bus.clone(),
            trace_id: self.trace_id.clone(),
            parent_span_id: self.parent_span_id.clone(),
        })
    }
}

/// Wrapper around any BaseTool that records tool call traces.
pub struct TracingTool<T: BaseTool> {
    inner: T,
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
    trace_id: String,
    parent_span_id: Option<String>,
}

impl<T: BaseTool> TracingTool<T> {
    pub fn new(
        inner: T,
        store: Arc<dyn TracingStore>,
        event_bus: EventBus,
        trace_id: String,
    ) -> Self {
        Self {
            inner,
            store,
            event_bus,
            trace_id,
            parent_span_id: None,
        }
    }

    pub fn with_parent_span(mut self, span_id: String) -> Self {
        self.parent_span_id = Some(span_id);
        self
    }
}

fn record_tool_span(
    store: &dyn TracingStore,
    event_bus: &EventBus,
    trace_id: &str,
    parent_span_id: &Option<String>,
    tool_name: &str,
    input: &JsonValue,
    result: &Result<JsonValue, langgraph_prebuilt::traits::ToolError>,
) {
    let span_id = Uuid::new_v4().to_string();
    let mut span = Span::new(
        span_id,
        trace_id.to_string(),
        parent_span_id.clone(),
        tool_name.to_string(),
        SpanType::ToolCall,
        input.clone(),
    );
    span.metadata.tool_name = Some(tool_name.to_string());

    match result {
        Ok(output) => {
            span.finish(output.clone(), SpanStatus::Success);
        }
        Err(e) => {
            span.finish(
                serde_json::json!({"error": e.to_string()}),
                SpanStatus::Error,
            );
        }
    }

    store.add_span(span.clone());
    event_bus.publish(crate::event_bus::TracingEvent::SpanCreated { span });
}

#[async_trait]
impl<T: BaseTool + 'static> BaseTool for TracingTool<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters(&self) -> Option<&JsonValue> {
        self.inner.parameters()
    }

    fn invoke(&self, args: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, langgraph_prebuilt::traits::ToolError> {
        let start = Instant::now();
        let result = self.inner.invoke(args, config);
        record_tool_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), args, &result);
        let _ = start;
        result
    }

    async fn ainvoke(&self, args: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, langgraph_prebuilt::traits::ToolError> {
        let start = Instant::now();
        let result = self.inner.ainvoke(args, config).await;
        record_tool_span(self.store.as_ref(), &self.event_bus, &self.trace_id, &self.parent_span_id, self.inner.name(), args, &result);
        let _ = start;
        result
    }

    fn to_tool_def(&self) -> ToolDef {
        self.inner.to_tool_def()
    }
}
