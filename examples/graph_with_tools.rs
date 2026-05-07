
use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph_derive::{tool, StateGraph};
use langgraph_prebuilt::{
    prepare_tools, stream_llm, tools_condition, BaseChatModel, Message,
    ToolNode,
};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Step 1: Define tools with 
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

#[tool("divide", "Divide a by b")]
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        return Err("Division by zero".to_string());
    }
    Ok(a / b)
}

// -------------------------------------------------------
// Step 2: Define state with #[derive(StateGraph)]
// -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
    #[channel]
    llm_calls: i64,
}

// -------------------------------------------------------
// Step 3: Build graph
// -------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Manual Graph with Tools (Simplified)");
    println!("========================================\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(Multiply::new()),
        Arc::new(Add::new()),
        Arc::new(Divide::new()),
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

    // LLM node — uses stream_llm for token-by-token streaming
    let model_clone = model_with_tools.clone();
    graph.add_node("llm_call", move |input: JsonValue, _config: RunnableConfig| {
        let model = model_clone.clone();
        async move {
            let mut result = stream_llm(
                model.as_ref(),
                &input,
                "You are a math assistant. You MUST use the provided tools to perform calculations. \
                 When the user asks for a calculation, call the appropriate tool (add, multiply, or divide) \
                 with the exact numbers provided. Do NOT make up numbers. \
                 After receiving the tool result, give the final answer.",
            )
            .await?;

            // Preserve custom state field
            let current = input.get("llm_calls").and_then(|v| v.as_i64()).unwrap_or(0);
            result.as_object_mut().unwrap().insert("llm_calls".to_string(), serde_json::json!(current + 1));
            Ok(result)
        }
    })?;

    // Tool node (prebuilt)
    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tool_node", tools_node)?;

    // Edges — uses conditional_edges! macro
    graph.add_edge(START, "llm_call")?;
    conditional_edges!(graph, "llm_call", tools_condition, "tools" => "tool_node", END => END)?;
    graph.add_edge("tool_node", "llm_call")?;

    // Compile
    let app = graph.compile()?;

    // -------------------------------------------------------
    // Step 4: Invoke with streaming 
    // -------------------------------------------------------

    let tests = [("Multiply 3 and 4.", 0_i64),
        ("Add 5 and 7.", 0),
        ("Divide 100 by 3.", 0)];

    for (i, (question, _)) in tests.iter().enumerate() {
        println!("--- Test {}: {} ---\n", i + 1, question);

        let input = serde_json::json!({
            "messages": [{"type": "human", "content": question}],
            "llm_calls": 0
        });

        let mut stream = app.astream(
            &input,
            &RunnableConfig::new(),
            vec![StreamMode::Custom, StreamMode::Updates],
        );

        use std::io::Write;
        while let Some(part) = stream.next().await {
            match part.mode {
                StreamMode::Custom => {
                    // Token-by-token typewriter output
                    if let Some(token_type) = part.data.get("type").and_then(|t| t.as_str()) {
                        if token_type == "token" {
                            if let Some(content) = part.data.get("content").and_then(|c| c.as_str()) {
                                print!("{}", content);
                                let _ = std::io::stdout().flush();
                            }
                        }
                    }
                }
                StreamMode::Updates => {
                    // Show tool calls from updates
                    if let Some(obj) = part.data.as_object() {
                        for (node_name, output) in obj {
                            if node_name == "tool_node" {
                                if let Some(messages) = output.get("messages").and_then(|m| m.as_array()) {
                                    for msg in messages {
                                        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                                        println!("\n  [tool result] {}", content);
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        println!("\n");
    }

    println!("========================================");
    println!("  All tests completed!");
    println!("========================================");

    Ok(())
}

