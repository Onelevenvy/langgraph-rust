use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph::graph::GraphError;
use langgraph::runnable::{Runnable, RunnableError};
use langgraph::channels::{BinaryOperatorAggregate, Channel};
use langgraph::constants::{START, END};
use langgraph::graph::StateGraph;

use crate::traits::{BaseChatModel, BaseTool, ToolDef};
use crate::types::{Message, add_messages};
use crate::tool_node::ToolNode;
use crate::tools_condition::tools_condition;

/// Configuration for creating a ReAct agent.
pub struct ReActAgentConfig {
    /// The system prompt to use.
    pub system_prompt: Option<String>,
    /// Maximum number of steps before the agent stops.
    pub max_steps: Option<usize>,
    /// Whether to handle tool errors gracefully.
    pub handle_tool_errors: bool,
}

impl Default for ReActAgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_steps: Some(25),
            handle_tool_errors: true,
        }
    }
}

/// A compiled ReAct agent graph.
///
/// This is a prebuilt graph that implements the ReAct (Reasoning + Acting) pattern:
/// 1. The model receives the conversation history and decides what to do
/// 2. If the model calls tools, they are executed and the results are added to the history
/// 3. The model sees the tool results and decides what to do next
/// 4. This continues until the model responds without tool calls (or max steps is reached)
pub struct ReActAgent {
    graph: Box<dyn Runnable>,
}

impl ReActAgent {
    /// Invoke the agent synchronously.
    pub fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        self.graph.invoke(input, config)
    }

    /// Invoke the agent asynchronously.
    pub async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        self.graph.ainvoke(input, config).await
    }
}

/// Reducer for messages channel: appends new messages to existing ones.
fn messages_reducer(current: &JsonValue, update: &JsonValue) -> JsonValue {
    add_messages(current.clone(), update.clone())
}

/// Create a ReAct agent with the given model and tools.
///
/// This builds a graph with the following structure:
/// ```text
/// START → agent → [tools_condition] → tools → agent (loop)
///                                → END
/// ```
///
/// # Arguments
/// * `model` - The chat model to use for generating responses
/// * `tools` - The tools available to the agent
/// * `config` - Optional configuration for the agent
///
/// # Returns
/// A compiled agent graph that can be invoked.
pub fn create_react_agent(
    model: Arc<dyn BaseChatModel>,
    tools: Vec<Arc<dyn BaseTool>>,
    config: Option<ReActAgentConfig>,
) -> Result<ReActAgent, GraphError> {
    let config = config.unwrap_or_default();

    // Get tool definitions for the model
    let tool_defs: Vec<ToolDef> = tools.iter().map(|t| t.to_tool_def()).collect();

    // Bind tools to the model (wrap in Arc for sharing across closures)
    let bound_model: Arc<dyn BaseChatModel> = Arc::from(model.bind_tools(tool_defs));

    // Create the ToolNode (wrapped in Arc for sharing across closures)
    let tool_node = Arc::new(
        ToolNode::new(tools).with_error_handling(config.handle_tool_errors)
    );

    // -------------------------------------------------------
    // Build graph: START → agent → [should_continue] → tools → agent (loop) → END
    // (same structure as Python create_react_agent)
    // -------------------------------------------------------

    // Create channels with reducers
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", messages_reducer)),
    );

    let mut graph = StateGraph::new(channels);

    // --- Agent node: calls the LLM ---
    let agent_model = bound_model;
    let system_prompt = config.system_prompt.clone();

    graph.add_node("agent", move |input: JsonValue, _config: RunnableConfig| {
        let model = agent_model.clone();
        let prompt = system_prompt.clone();
        async move {
            let messages = match input.get("messages") {
                Some(JsonValue::Array(arr)) => arr.clone(),
                _ => vec![],
            };

            let mut typed_messages: Vec<Message> = Vec::new();

            if let Some(ref p) = prompt {
                typed_messages.push(Message::system(p.clone()));
            }

            for msg in &messages {
                if let Ok(m) = serde_json::from_value::<Message>(msg.clone()) {
                    typed_messages.push(m);
                }
            }

            let response = model.invoke(&typed_messages, &RunnableConfig::new())
                .map_err(|e| RunnableError::Node(e.to_string()))?;
            let response_json = serde_json::to_value(response)
                .map_err(|e: serde_json::Error| RunnableError::Node(e.to_string()))?;

            Ok(serde_json::json!({
                "messages": [response_json]
            }))
        }
    })?;

    // --- Tools node: executes tool calls ---
    let tools_arc = tool_node.clone();
    graph.add_node("tools", move |input: JsonValue, config: RunnableConfig| {
        let tn = tools_arc.clone();
        async move {
            tn.ainvoke(&input, &config).await
        }
    })?;

    // --- Conditional edge: agent → tools or END ---
    graph.add_conditional_edges(
        "agent",
        |input: JsonValue, _config: RunnableConfig| async move {
            let route = tools_condition(&input);
            Ok(JsonValue::String(route))
        },
        Some({
            let mut map = HashMap::new();
            map.insert("tools".to_string(), "tools".to_string());
            map.insert(END.to_string(), END.to_string());
            map
        }),
    )?;

    // --- Edge: tools → agent (loop back) ---
    graph.add_edge("tools", "agent")?;

    // --- Entry point ---
    graph.add_edge(START, "agent")?;

    // --- Compile ---
    let mut builder = graph.compile_builder();
    if let Some(steps) = config.max_steps {
        builder = builder.recursion_limit(steps as u64);
    }
    let compiled = builder.build()?;

    Ok(ReActAgent {
        graph: Box::new(compiled),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_merge_state() {
        // The reducer receives raw channel values (arrays), not objects with "messages" key
        let current = serde_json::json!([
            {"type": "human", "content": "Hi"}
        ]);
        let update = serde_json::json!([
            {"type": "ai", "content": "Hello"}
        ]);

        let merged = messages_reducer(&current, &update);
        let messages = merged.as_array().unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_merge_state_new_key() {
        let current = serde_json::json!({
            "messages": []
        });
        let update = serde_json::json!({
            "result": "done"
        });

        let _merged = messages_reducer(&current, &update);
        // add_messages merges the messages arrays
        // "result" is not messages, so it gets appended as a message
    }
}
