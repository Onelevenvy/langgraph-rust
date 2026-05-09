use std::io::{self, Write};
use std::sync::Arc;
use serde_json::{json, Value as JsonValue};
use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph_checkpoint::checkpoint::memory::InMemorySaver;
use langgraph_checkpoint::config::RunnableConfigExt;
use langgraph_derive::{tool, StateGraph, Traceable};
use langgraph_prebuilt::{
    prepare_tools, stream_llm, stream_and_print, tools_condition, BaseChatModel, Message,
    ToolNode,
};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
use serde::{Deserialize, Serialize};

// Tracing imports - only TracingChatModel needed for LLM wrapping
use langgraph_tracing::TracingChatModel;

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Tools
// -------------------------------------------------------

#[tool("multiply", "Multiply two integers a and b")]
fn multiply(a: i64, b: i64) -> Result<i64, String> {
    a.checked_mul(b).ok_or_else(|| "Multiplication overflow".to_string())
}

#[tool("add", "Add two integers a and b")]
fn add(a: i64, b: i64) -> Result<i64, String> {
    a.checked_add(b).ok_or_else(|| "Addition overflow".to_string())
}

#[tool("get_weather", "Get the current weather for a location")]
fn get_weather(location: String) -> Result<String, String> {
    Ok(format!("Weather for {}: sunny, 22°C", location))
}

// -------------------------------------------------------
// State
// -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph, Traceable)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Interactive Chat with Real-time Tracing");
    println!("========================================");
    println!("  1. Starting Tracing Server...");

    // One-liner tracing setup via #[derive(Traceable)]
    let mut tracing = GraphState::tracing_context();
    tracing.start_server("127.0.0.1:3333");

    println!("  2. Tracing UI available at http://127.0.0.1:3333");
    println!("  3. Type 'quit' to exit.\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(Multiply::new()),
        Arc::new(Add::new()),
        Arc::new(GetWeather::new()),
    ]);

    // Create base model
    let (api_key, api_base, model_name) = load_openai_config();
    let base_model = Arc::new(OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.0),
        ..Default::default()
    }));

    // Build graph
    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    // Node: LLM Call with dynamic tracing wrapper
    let model_arc = base_model.clone();
    let tool_defs = prepared.tool_defs.clone();

    // Access tracing internals for the node closure
    let store = tracing.store().clone();
    let bus = tracing.event_bus().clone();

    graph.add_node("llm_call", move |input: JsonValue, config: RunnableConfig| {
        let model = model_arc.clone();
        let store = store.clone();
        let bus = bus.clone();
        let tool_defs = tool_defs.clone();

        async move {
            let trace_id = config.get_configurable()
                .and_then(|c| c.get("trace_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();

            let tracing_model = TracingChatModel::new(
                model.bind_tools(tool_defs),
                store,
                bus,
                trace_id
            );

            stream_llm(
                &tracing_model,
                &input,
                "You are a helpful assistant with math and weather tools.",
            )
            .await
        }
    })?;

    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tool_node", tools_node)?;

    graph.add_edge(START, "llm_call")?;
    conditional_edges!(graph, "llm_call", tools_condition, "tools" => "tool_node", END => END)?;
    graph.add_edge("tool_node", "llm_call")?;

    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    // Interactive loop
    let stdin = io::stdin();
    let mut turn = 0u32;

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 { break; }
        let input_line = input_line.trim();

        if input_line.eq_ignore_ascii_case("quit") || input_line.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }
        if input_line.is_empty() { continue; }

        turn += 1;
        println!("\n--- Turn {} ---", turn);

        let input = json!({
            "messages": [{"type": "human", "content": input_line}]
        });

        // Single call: trace lifecycle is automatic
        let mut config = RunnableConfig::new();
        config.insert("configurable".to_string(), json!({
            "thread_id": "interactive-session"
        }));

        let collected_text = tracing.run_with_tracing(
            "interactive_chat_turn",
            input.clone(),
            config,
            |config| {
                let app = &app;
                async move {
                    let mut stream = app.astream(&input, &config, vec![StreamMode::Custom, StreamMode::Updates]);
                    print!("Assistant: ");
                    let text = stream_and_print(&mut stream, false).await;
                    println!("\n");
                    json!({"messages": [{"type": "ai", "content": text}]})
                }
            },
        ).await;

        let _ = collected_text;
    }

    Ok(())
}
