use langgraph::prelude::*;
use langgraph::langgraph_state;
use langgraph::prebuilt::Message;

#[langgraph_state]
#[derive(Debug)]
struct AgentState {
    #[channel(messages)]
    messages: Vec<Message>,
    step_count: i64,  // defaults to LastValue
}

/// Agent node: generates a response and increments step count.
async fn agent(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    Ok(serde_json::json!({
        "messages": [{"type": "ai", "content": "Hello from the agent!"}],
        "step_count": 1
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  StateGraph Derive Macro Example");
    println!("========================================\n");

    // Create channels from the derived struct (one line!)
    let channels = AgentState::create_channels();
    let mut graph = StateGraph::new(channels);

    // Add nodes (passing standalone function, just like Python!)
    graph.add_node("agent", agent)?;

    // Add edges
    graph.add_edge(START, "agent")?;
    graph.add_edge("agent", END)?;

    // Compile with defaults (one line!)
    let compiled = graph.compile()?;

    // Invoke
    let result = compiled.ainvoke(
        &serde_json::json!({"step_count": 0}),
        &RunnableConfig::new(),
    ).await?;

    println!("Result:");
    println!("{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}
