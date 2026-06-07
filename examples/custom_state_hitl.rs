use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::checkpoint::InMemorySaver;
use langgraph::{langgraph_state, tool};
use langgraph::prebuilt::{
    invoke_llm, prepare_tools, tools_condition, BaseChatModel, Message, ToolError, ToolNode,
};
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
// Step 1: Define tools
// -------------------------------------------------------

#[tool(
    "human_assistance",
    "Request assistance from a human. Use this when you need human review or correction of information."
)]
fn human_assistance(name: String, birthday: String) -> Result<String, ToolError> {
    // Pause execution and wait for human input
    let human_response = interrupt(serde_json::json!({
        "question": "Is this correct?",
        "name": name,
        "birthday": birthday,
    }))?;

    // Determine verified values based on human response
    let (verified_name, verified_birthday, response) =
        if let Some(correct) = human_response.get("correct").and_then(|v| v.as_str()) {
            if correct.to_lowercase().starts_with('y') {
                (name.clone(), birthday.clone(), "Correct".to_string())
            } else {
                let corrected_name = human_response
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&name)
                    .to_string();
                let corrected_birthday = human_response
                    .get("birthday")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&birthday)
                    .to_string();
                let msg = format!("Made a correction: {}", human_response);
                (corrected_name, corrected_birthday, msg)
            }
        } else {
            // If no "correct" field, treat the whole response as the correction
            let corrected_name = human_response
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or(&name)
                .to_string();
            let corrected_birthday = human_response
                .get("birthday")
                .and_then(|v| v.as_str())
                .unwrap_or(&birthday)
                .to_string();
            let msg = format!("Made a correction: {}", human_response);
            (corrected_name, corrected_birthday, msg)
        };

    // Return a Command that updates custom state fields.
    // The ToolNode will:
    // 1. Extract the ToolMessage from update.messages (fixing up tool_call_id)
    // 2. Apply name and birthday updates to their respective channels
    let cmd = Command {
        graph: None,
        resume: None,
        goto: vec![],
        update: Some(serde_json::json!({
            "name": verified_name,
            "birthday": verified_birthday,
            "messages": [Message::tool_result("__placeholder__", response)],
        })),
    };

    Ok(serde_json::to_string(&cmd).unwrap_or_else(|_| "{}".to_string()))
}

/// Mock search tool (replaces TavilySearch in the Python example).
#[tool("search", "Search for information about a topic.")]
fn search(query: String) -> Result<String, String> {
    // Mock search results
    let results = match query.to_lowercase().as_str() {
        q if q.contains("langgraph") => {
            r#"[{"url": "https://blog.langchain.dev/langgraph-cloud/", "content": "LangGraph Platform was announced on June 27, 2024. LangGraph had been in development before this."}]"#
        }
        q if q.contains("rust") => {
            r#"[{"url": "https://www.rust-lang.org/", "content": "Rust 1.0 was released on May 15, 2015."}]"#
        }
        _ => r#"[{"url": "https://example.com", "content": "No relevant results found."}]"#,
    };
    Ok(results.to_string())
}

// -------------------------------------------------------
// Step 2: Define custom state
// -------------------------------------------------------

// ...
#[langgraph_state]
#[derive(Debug)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
    /// Custom field: entity name being researched
    name: String,
    /// Custom field: birthday/release date of the entity
    birthday: String,
}

// -------------------------------------------------------
// Step 3: Build and run
// -------------------------------------------------------

const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that researches information about entities. \
    Use the search tool to find information, then use the human_assistance tool \
    to get human review of your findings. When calling human_assistance, \
    provide your suggested name and birthday in the tool arguments.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Custom State + HITL Demo");
    println!("========================================\n");

    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(HumanAssistance::new()),
        Arc::new(Search::new()),
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

    // Chatbot node
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
    let checkpointer = Arc::new(InMemorySaver::new());
    let app = graph.compile_builder().checkpointer(checkpointer).build()?;

    // -------------------------------------------------------
    // Step 4: Run the demo
    // -------------------------------------------------------

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({
            "thread_id": "custom-state-demo-1"
        }),
    );

    println!("--- Step 1: Initial query ---\n");
    println!(
        "User: Can you look up when LangGraph was released? \
              When you have the answer, use the human_assistance tool for review.\n"
    );

    let input = serde_json::json!({
        "messages": [{
            "type": "human",
            "content": "Can you look up when LangGraph was released? \
                        When you have the answer, use the human_assistance tool for review."
        }]
    });

    // First invocation - triggers search, then human_assistance, then interrupt
    let result = app.ainvoke(&input, &config).await?;

    println!("--- Graph paused (interrupt occurred) ---\n");
    print_messages(&result);

    // -------------------------------------------------------
    // Step 5: Check state with get_state
    // -------------------------------------------------------

    println!("\n--- Step 2: Check state with get_state ---\n");

    let snapshot = app.get_state(&config)?;
    println!("snapshot.next = {:?}", snapshot.next);
    println!(
        "snapshot.values.name = {}",
        snapshot
            .values
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)")
    );
    println!(
        "snapshot.values.birthday = {}",
        snapshot
            .values
            .get("birthday")
            .and_then(|v| v.as_str())
            .unwrap_or("(empty)")
    );
    println!("snapshot.interrupts = {:?}", snapshot.interrupts);

    // -------------------------------------------------------
    // Step 6: Resume with human correction
    // -------------------------------------------------------

    println!("\n--- Step 3: Resume with human correction ---\n");
    println!("Human: The name is 'LangGraph' and the birthday is 'Jan 17, 2024'\n");

    let resume_command = Command::resume(serde_json::json!({
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    }));

    let result = app
        .ainvoke(&serde_json::to_value(&resume_command)?, &config)
        .await?;

    println!("--- Result after resume ---\n");
    print_messages(&result);

    // -------------------------------------------------------
    // Step 7: Verify custom state via get_state
    // -------------------------------------------------------

    println!("\n--- Step 4: Verify custom state ---\n");

    let snapshot = app.get_state(&config)?;
    let name = snapshot
        .values
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("(empty)");
    let birthday = snapshot
        .values
        .get("birthday")
        .and_then(|v| v.as_str())
        .unwrap_or("(empty)");

    println!("{{'name': '{}', 'birthday': '{}'}}", name, birthday);

    // -------------------------------------------------------
    // Step 8: Manually update state
    // -------------------------------------------------------

    println!("\n--- Step 5: Manually update state ---\n");

    app.update_state(
        &config,
        &serde_json::json!({
            "name": "LangGraph (library)"
        }),
    )?;

    let snapshot = app.get_state(&config)?;
    let name = snapshot
        .values
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("(empty)");
    let birthday = snapshot
        .values
        .get("birthday")
        .and_then(|v| v.as_str())
        .unwrap_or("(empty)");

    println!("After update_state:");
    println!("{{'name': '{}', 'birthday': '{}'}}", name, birthday);

    println!("\n========================================");
    println!("  Demo completed!");
    println!("========================================");

    Ok(())
}

fn print_messages(output: &JsonValue) {
    if let Some(messages) = output.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if let Ok(m) = serde_json::from_value::<Message>(msg.clone()) {
                println!("{}", m);
            }
        }
    }
}
