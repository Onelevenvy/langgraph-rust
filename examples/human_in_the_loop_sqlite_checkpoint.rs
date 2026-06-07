use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::sqlite::SqliteSaver;
use langgraph::{langgraph_state, tool};
use langgraph::prebuilt::{
    invoke_llm, prepare_tools, tools_condition, BaseChatModel, Message, ToolError, ToolNode,
};
use langgraph::providers::openai::{OpenAIModel, OpenAIModelConfig};
use serde_json::Value as JsonValue;
use std::sync::Arc;

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "Pro/deepseek-ai/DeepSeek-V3.2".to_string());

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
#[langgraph_state]
#[derive(Debug)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

// -------------------------------------------------------
// Step 3: Build graph and run demo
// -------------------------------------------------------

const SYSTEM_PROMPT: &str = "You are a helpful assistant with access to tools. \
    Use the human_assistance tool when the user needs expert guidance. \
    IMPORTANT: After receiving the result from the human_assistance tool, you MUST immediately synthesize the expert's advice and present it to the user. Do NOT ask clarifying questions or call the human_assistance tool multiple times for the same user request. \
    Use the get_weather tool for weather queries.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Robust Human-in-the-Loop Demo");
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

    // LLM node
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

    // Edges
    graph.add_edge(START, "chatbot")?;
    conditional_edges!(graph, "chatbot", tools_condition, "tools" => "tools", END => END)?;
    graph.add_edge("tools", "chatbot")?;

    // Compile with checkpointer
    let saver = SqliteSaver::from_conn_string("sqlite:checkpoints.db").await?;
    saver.setup().await?; 
    let checkpointer = Arc::new(saver);
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    // -------------------------------------------------------
    // Step 4: Dynamic Execution Loop
    // -------------------------------------------------------
    let thread_id = uuid::Uuid::new_v4().to_string();
    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({
            "thread_id": thread_id
        }),
    );

    // Initial query
    let mut current_input = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "I need some expert guidance for building an AI agent. Could you request assistance for me?"
        }]
    });

    let mut step_count = 1;
    let mut seen_messages = 0; // Tracks printed messages to avoid duplicate console output

    loop {
        println!("----------------------------------------");
        println!("▶ Execution Step {}", step_count);
        println!("----------------------------------------");

        // Execute the graph (can be cold start, new message input, or resume)
        let result = app.ainvoke(&current_input, &config).await?;

        // Extract and print only the newly generated messages in this step
        let messages_array = result.get("messages").and_then(|m| m.as_array()).unwrap();
        for msg_val in messages_array.iter().skip(seen_messages) {
            if let Ok(m) = serde_json::from_value::<Message>(msg_val.clone()) {
                println!("{}", m);
            }
        }
        seen_messages = messages_array.len();

        // Analyze the last message to determine the next action
        let last_msg_val = messages_array.last().unwrap();
        let last_msg: Message = serde_json::from_value(last_msg_val.clone())?;

        match last_msg {
            Message::Ai { tool_calls, .. } => {
                if tool_calls.is_empty() {
                    // The model output plain text. This means either:
                    // 1. It summarized the tool result and finished the task -> break
                    // 2. It asked a clarifying question without calling a tool -> reply

                    // Check if a Tool message exists in the conversation history
                    let has_used_tool = messages_array.iter().any(|m| {
                        matches!(serde_json::from_value::<Message>(m.clone()), Ok(Message::Tool { .. }))
                    });

                    if has_used_tool {
                        println!("\n[System 🤖] -> Expert advice synthesized and presented to the user. Demo completed! 🎉");
                        break;
                    } else {
                        println!("\n[System 🤖] -> The model asked a clarifying question without calling the tool.");
                        println!("[System 🤖] -> Automatically simulating the user to provide a follow-up answer...");
                        
                        // Provide a follow-up answer forcing the model to use the tool
                        current_input = serde_json::json!({
                            "messages": [{
                                "type": "human",
                                "content": "I am building a customer support chatbot in Rust. I just need you to ping the human expert for their architectural recommendation. Please invoke the tool now."
                            }]
                        });
                    }
                } else {
                    // The model invoked a tool.
                    // Since 'human_assistance' uses interrupt(), the graph is currently paused.
                    println!("\n[System 🤖] -> The model invoked a tool! Graph execution is paused (Interrupt).");
                    println!("[System 🤖] -> Automatically simulating the human expert to inject the response via Command::resume...");
                    
                    // Inject data using Command::resume to unpause the graph
                    let resume_command = Command::resume(serde_json::json!({
                        "data": "Expert advice: Use LangGraph-Rust for state machine orchestration and keep your tools modular!"
                    }));
                    current_input = serde_json::to_value(&resume_command)?;
                }
            },
            Message::Tool { .. } => {
                // If the last message is unexpectedly a Tool message, continue with null input
                current_input = serde_json::json!(null);
            },
            _ => {
                println!("\n[System 🤖] -> Unexpected graph state encountered. Terminating.");
                break;
            }
        }

        step_count += 1;
        if step_count > 10 {
            println!("\n[System 🤖] -> Maximum iteration limit (10) reached. Terminating to prevent infinite loop.");
            break;
        }
        
        println!("\n"); // Padding between steps
    }

    Ok(())
}