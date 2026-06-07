use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::checkpoint::InMemorySaver;
use langgraph::{langgraph_state, tool};
use langgraph::prebuilt::{
    invoke_llm, prepare_tools, print_result, tools_condition, BaseChatModel, Message, ToolNode,
};
use langgraph::providers::openai::{OpenAIModel, OpenAIModelConfig};

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name =
        std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Step 1: Define state
// -------------------------------------------------------
#[langgraph_state]
#[derive(Debug)]
struct State {
    #[channel(messages)]
    messages: Vec<Message>,
}

// -------------------------------------------------------
// Step 2: Define tools with #[tool] macro (Simplified)
// -------------------------------------------------------

#[tool("search", "Search for information about a topic.")]
fn search(query: String) -> Result<String, String> {
    // 模拟搜索结果
    Ok(format!(
        r#"[{{"url": "https://example.com/{}", "content": "Mock search result for: {}"}}]"#,
        query, query
    ))
}

// -------------------------------------------------------
// Step 3: Build graph
// -------------------------------------------------------

fn build_graph(
    model: Arc<dyn BaseChatModel>,
    tools: Vec<Arc<dyn langgraph::prebuilt::traits::BaseTool>>,
) -> Result<CompiledStateGraph, Box<dyn std::error::Error>> {
    let prepared = prepare_tools(tools);
    let model_with_tools: Arc<dyn BaseChatModel> = model.bind_tools(prepared.tool_defs).into();

    let channels = State::create_channels();
    let mut graph = StateGraph::new(channels);

    // Chatbot node
    let model_clone = model_with_tools.clone();
    graph.add_node(
        "chatbot",
        move |input: JsonValue, _config: RunnableConfig| {
            let model = model_clone.clone();
            async move { invoke_llm(model.as_ref(), &input, "You are a helpful assistant.") }
        },
    )?;

    // Tool node
    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tools", tools_node)?;

    // Edges
    graph.add_edge(START, "chatbot")?;
    conditional_edges!(graph, "chatbot", tools_condition, "tools" => "tools", END => END)?;
    graph.add_edge("tools", "chatbot")?;

    // Compile with checkpointer
    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    Ok(app)
}

// -------------------------------------------------------
// Step 4: Run the time travel demo
// -------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Time Travel Demo");
    println!("========================================\n");

    // Create model
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.7),
        ..Default::default()
    });

    // Build graph with the simplified #[tool] macro
    // `fn search` is automatically compiled into a `Search` struct by the macro
    let tools: Vec<Arc<dyn langgraph::prebuilt::traits::BaseTool>> = vec![Arc::new(Search::new())];
    let app = build_graph(Arc::new(model), tools)?;

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({"thread_id": "time-travel-demo"}),
    );

    // -------------------------------------------------------
    // Add steps - first conversation turn
    // -------------------------------------------------------
    println!("--- Step 1: First user message ---\n");
    println!("User: I'm learning LangGraph. Could you do some research on it for me?\n");

    let input1 = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "I'm learning LangGraph. Could you do some research on it for me?"
        }]
    });

    let result = app.ainvoke(&input1, &config).await?;
    print_result(&result);

    // -------------------------------------------------------
    // Add steps - second conversation turn
    // -------------------------------------------------------
    println!("\n--- Step 2: Second user message ---\n");
    println!("User: Ya that's helpful. Maybe I'll build an autonomous agent with it!\n");

    let input2 = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "Ya that's helpful. Maybe I'll build an autonomous agent with it!"
        }]
    });

    let result = app.ainvoke(&input2, &config).await?;
    print_result(&result);

    // -------------------------------------------------------
    // Replay full state history
    // -------------------------------------------------------
    println!("\n--- Step 3: State History ---\n");

    let history = app.get_state_history(&config)?;

    for (i, snapshot) in history.iter().enumerate() {
        let msg_count = snapshot
            .values
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        println!(
            "[{}] Num Messages: {}, Next: {:?}",
            i, msg_count, snapshot.next
        );
        println!("{}", "-".repeat(80));
    }

    // -------------------------------------------------------
    // Fork from a specific checkpoint
    // -------------------------------------------------------
    let fork_index = history
        .iter()
        .position(|s| !s.next.is_empty())
        .unwrap_or(0);
    let to_replay = &history[fork_index];

    let msg_count = to_replay
        .values
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    println!("\n--- Step 4: Fork from checkpoint [{}] ({} messages) ---\n", fork_index, msg_count);
    println!("Next nodes: {:?}\n", to_replay.next);

    // Resume execution from this checkpoint
    let result = app.ainvoke(&JsonValue::Null, &to_replay.config).await?;

    println!("Forked execution result:");
    print_result(&result);

    println!("\n========================================");
    println!("  Time Travel Demo completed!");
    println!("========================================");

    Ok(())
}
