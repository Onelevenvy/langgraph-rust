
use std::io::{self, Write};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph_checkpoint::checkpoint::memory::InMemorySaver;
use langgraph_derive::{tool, StateGraph};
use langgraph_prebuilt::{
    prepare_tools, stream_llm, stream_and_print, tools_condition, BaseChatModel, Message,
    ToolNode,
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
// Tools
// -------------------------------------------------------

#[tool("multiply", "Multiply two integers a and b")]
fn multiply(a: i64, b: i64) -> Result<i64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "Multiplication overflow".to_string())
}

#[tool("add", "Add two integers a and b")]
fn add(a: i64, b: i64) -> Result<i64, String> {
    a.checked_add(b)
        .ok_or_else(|| "Addition overflow".to_string())
}

#[tool(
    "get_weather",
    "Get the current weather for a given location. Returns temperature, humidity and conditions."
)]
fn get_weather(location: String) -> Result<String, String> {
    // 模拟天气数据
    Ok(format!(
        "Weather for {}: sunny, 22°C, humidity 45%, wind 10km/h",
        location
    ))
}

// -------------------------------------------------------
// State
// -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

// -------------------------------------------------------
// Build graph
// -------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Interactive Chat with Tools");
    println!("========================================");
    println!("  Type 'quit' to exit.\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(Multiply::new()),
        Arc::new(Add::new()),
        Arc::new(GetWeather::new()),
    ]);

    // Create model and bind tools
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.0),
        ..Default::default()
    });
    let model_with_tools: Arc<dyn BaseChatModel> = model.bind_tools(prepared.tool_defs).into();

    // Build graph
    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    let model_clone = model_with_tools.clone();
    graph.add_node("llm_call", move |input: JsonValue, _config: RunnableConfig| {
        let model = model_clone.clone();
        async move {
            stream_llm(
                model.as_ref(),
                &input,
                "You are a math assistant.",
            )
            .await
        }
    })?;

    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tool_node", tools_node)?;

    graph.add_edge(START, "llm_call")?;
    conditional_edges!(graph, "llm_call", tools_condition, "tools" => "tool_node", END => END)?;
    graph.add_edge("tool_node", "llm_call")?;

    // Compile with checkpointer for multi-turn conversation
    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({"thread_id": "interactive-session"}),
    );

    // -------------------------------------------------------
    // Interactive loop
    // -------------------------------------------------------

    let stdin = io::stdin();
    let mut turn = 0u32;

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 {
            break; // EOF
        }
        let input_line = input_line.trim();

        if input_line.eq_ignore_ascii_case("quit") || input_line.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        if input_line.is_empty() {
            continue;
        }

        turn += 1;
        println!("\n--- Turn {} ---\n", turn);

        let input = serde_json::json!({
            "messages": [{"type": "human", "content": input_line}]
        });

        let mut stream = app.astream(
            &input,
            &config,
            vec![StreamMode::Custom, StreamMode::Updates],
        );

        print!("Assistant: ");
        let _ = stream_and_print(&mut stream, false).await;
        println!("\n");
    }

    Ok(())
}
