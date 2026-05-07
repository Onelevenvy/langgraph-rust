
use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph_checkpoint::checkpoint::memory::InMemorySaver;
use langgraph_derive::{tool, StateGraph};
use langgraph_prebuilt::{
    invoke_llm, prepare_tools, tools_condition, BaseChatModel, Message, ToolError, ToolNode,
};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
use serde::{Deserialize, Serialize};

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Step 1: Define tools with #[tool] macro
// -------------------------------------------------------

/// Request assistance from a human. Uses interrupt() to pause execution.
#[tool(
    "human_assistance",
    "Request assistance from a human. Use this when you need human expertise or approval."
)]
fn human_assistance(query: String) -> Result<String, ToolError> {
    let human_response = interrupt(serde_json::json!({
        "query": query,
        "message": "Please provide your response to the human assistance request."
    }))?;

    if let Some(data) = human_response.get("data").and_then(|v| v.as_str()) {
        Ok(data.to_string())
    } else {
        Ok(human_response.to_string())
    }
}

/// Get the current weather for a location (mock implementation).
#[tool("get_weather", "Get the current weather for a given location.")]
fn get_weather(location: String) -> Result<String, String> {
    Ok(format!(
        "Weather for {}: sunny, 22°C, humidity 45%, wind 10km/h",
        location
    ))
}

// -------------------------------------------------------
// Step 2: Define state with #[derive(StateGraph)]
// -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

// -------------------------------------------------------
// Step 3: Build graph and run demo
// -------------------------------------------------------

const SYSTEM_PROMPT: &str = "You are a helpful assistant with access to tools. \
    Use the human_assistance tool when the user needs expert guidance. \
    Use the get_weather tool for weather queries. \
    After receiving tool results, provide a helpful response.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Human-in-the-Loop Demo");
    println!("========================================\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(HumanAssistance::new()),
        Arc::new(GetWeather::new()),
    ]);

    // Create model and bind tools
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.7),
        ..Default::default()
    });
    let model_with_tools: Arc<dyn BaseChatModel> = model.bind_tools(prepared.tool_defs).into();

    // Build graph
    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    // LLM node — 3 lines instead of 25
    let model_clone = model_with_tools.clone();
    graph.add_node(
        "chatbot",
        move |input: JsonValue, _config: RunnableConfig| {
            let model = model_clone.clone();
            async move { invoke_llm(model.as_ref(), &input, SYSTEM_PROMPT) }
        },
    )?;

    // Tool node
    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tools", tools_node)?;

    // Edges — uses conditional_edges! macro
    graph.add_edge(START, "chatbot")?;
    conditional_edges!(graph, "chatbot", tools_condition, "tools" => "tools", END => END)?;
    graph.add_edge("tools", "chatbot")?;

    // Compile with checkpointer
    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    // -------------------------------------------------------
    // Step 4: Run the demo
    // -------------------------------------------------------

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({
            "thread_id": "demo-thread-1"
        }),
    );

    println!("--- Step 1: Initial query ---\n");
    println!("User: I need some expert guidance for building an AI agent.\n");

    let input = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "I need some expert guidance for building an AI agent. Could you request assistance for me?"
        }]
    });

    // First invocation - triggers human_assistance tool and interrupt
    let result = app.ainvoke(&input, &config).await?;

    println!("--- Graph paused (interrupt occurred) ---\n");
    println!("[DEBUG] Raw state after first invocation:");
    println!(
        "{}",
        serde_json::to_string_pretty(&result).unwrap_or_default()
    );
    println!();
    print_result(&result);

    println!("\n--- Step 2: Resume with human response ---\n");
    println!("Human: We recommend using LangGraph for building AI agents!\n");

    // Resume with human input
    let resume_command = Command::resume(serde_json::json!({
        "data": "We, the experts are here to help! We'd recommend you check out LangGraph  or langchain to build your agent."
    }));

    let result = app
        .ainvoke(&serde_json::to_value(&resume_command)?, &config)
        .await?;

    println!("--- Final result ---\n");
    println!("[DEBUG] Raw state after resume:");
    println!(
        "{}",
        serde_json::to_string_pretty(&result).unwrap_or_default()
    );
    println!();
    print_result(&result);

    println!("\n========================================");
    println!("  Demo completed!");
    println!("========================================");

    Ok(())
}

fn print_result(output: &JsonValue) {
    if let Some(messages) = output.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let msg_type = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match msg_type {
                "human" => {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        println!("User:      {}", content);
                    }
                }
                "ai" => {
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let has_tools = msg
                        .get("tool_calls")
                        .and_then(|tc| tc.as_array())
                        .map(|arr| !arr.is_empty())
                        .unwrap_or(false);
                    if has_tools {
                        println!("Assistant: [calling tools...]");
                        if let Some(calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                            for tc in calls {
                                let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                                let args =
                                    tc.get("args").map(|a| a.to_string()).unwrap_or_default();
                                println!("  -> {}({})", name, args);
                            }
                        }
                    } else if !content.is_empty() {
                        println!("Assistant: {}", content);
                    }
                }
                "tool" => {
                    let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    let tool_id = msg
                        .get("tool_call_id")
                        .and_then(|t| t.as_str())
                        .unwrap_or("?");
                    println!("Tool [{}]: {}", tool_id, content);
                }
                _ => {}
            }
        }
    }
}
