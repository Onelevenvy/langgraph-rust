use std::sync::Arc;

use serde_json::Value as JsonValue;
use tokio_stream::StreamExt;

use dotenvy::dotenv;
use langgraph::config::get_stream_writer;
use langgraph::prelude::*;
use langgraph::{langgraph_state, tool};
use langgraph::prebuilt::{prepare_tools, stream_llm, tools_condition, BaseChatModel, Message, ToolNode};
use langgraph::providers::openai::{OpenAIModel, OpenAIModelConfig};


fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// Define a tool that streams progress updates
// -------------------------------------------------------

#[tool("research", "Research a topic and stream progress updates")]
fn research(topic: String) -> Result<String, String> {
    // Get the stream writer to send custom chunks
    let writer = get_stream_writer();

    // Simulate research steps with streaming progress
    let steps = vec![
        format!("Searching for '{}'...", topic),
        format!("Found 10 results for '{}'...", topic),
        "Analyzing content...".to_string(),
        "Summarizing findings...".to_string(),
        "Research complete!".to_string(),
    ];

    let mut result = String::new();
    for step in &steps {
        if let Some(ref w) = writer {
            let _ = w.try_send(serde_json::json!({
                "status": "progress",
                "message": step
            }));
        }
        result.push_str(step);
        result.push('\n');
    }

    Ok(format!(
        "Research results for '{}': Found comprehensive information. \
         Key findings: This is a fascinating area with many applications.",
        topic
    ))
}

// -------------------------------------------------------
// Define state
// -------------------------------------------------------
#[langgraph_state]
#[derive(Debug)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

// -------------------------------------------------------
// Build and run
// -------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Streaming Demo: Typewriter Effect");
    println!("========================================\n");

    // Prepare tools
    let prepared = prepare_tools(vec![Arc::new(Research::new())]);

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

    // LLM node — uses stream_llm for token-by-token streaming
    let model_clone = model_with_tools.clone();
    graph.add_node("chatbot", move |input: JsonValue, _config: RunnableConfig| {
        let model = model_clone.clone();
        async move {
            stream_llm(
                model.as_ref(),
                &input,
                "You are a research assistant. Use the research tool when the user asks about a topic. \
                 After receiving research results, provide a helpful summary.",
            )
            .await
        }
    })?;

    // Tool node
    let tools_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tools", tools_node)?;

    // Edges
    graph.add_edge(START, "chatbot")?;
    conditional_edges!(graph, "chatbot", tools_condition, "tools" => "tools", END => END)?;
    graph.add_edge("tools", "chatbot")?;

    // Compile
    let app = graph.compile()?;

    // -------------------------------------------------------
    // Stream with Custom + Updates modes
    // -------------------------------------------------------

    println!("User: Research the topic 'Rust async programming'\n");

    let input = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "Research the topic 'Rust async programming'"
        }]
    });

    let mut stream = app.astream(
        &input,
        &RunnableConfig::new(),
        vec![StreamMode::Custom, StreamMode::Updates],
    );

    println!("Streaming output:");
    println!("─────────────────");
    while let Some(part) = stream.next().await {
        match part.mode {
            StreamMode::Custom => {
                // Token chunks from stream_llm (typewriter effect)
                if let Some(token_type) = part.data.get("type").and_then(|t| t.as_str()) {
                    if token_type == "token" {
                        if let Some(content) = part.data.get("content").and_then(|c| c.as_str()) {
                            print!("{}", content);
                            // Flush to show tokens immediately
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                            continue;
                        }
                    }
                }
                // Progress updates from tools
                if let Some(message) = part
                    .data
                    .get("message")
                    .and_then(|m: &JsonValue| m.as_str())
                {
                    println!("\n  [tool] {}", message);
                }
            }
            StreamMode::Updates => {
                // Node output updates — just show node name, content was already streamed
                if let Some(obj) = part.data.as_object() {
                    for (node_name, _output) in obj {
                        println!("\n  [update] Node '{}' completed", node_name);
                    }
                }
            }
            _ => {}
        }
    }

    println!("\n========================================");
    println!("  Demo completed!");
    println!("========================================");

    Ok(())
}
