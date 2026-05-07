use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::runnable::{RunnableCallable, RunnableError};
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::{BaseChatModel, Message};
use langgraph_providers::openai::OpenAIModel;

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

/// Simple chat: create model → build chat_node → call LLM.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        // 1. Create the OpenAI-compatible model
        let (api_key, api_base, model_name) = load_openai_config();
        let model = OpenAIModel::new(langgraph_providers::openai::OpenAIModelConfig {
            model: model_name.clone(),
            api_key,
            api_base,
            temperature: Some(0.7),
            ..Default::default()
        });

        // 2. Wrap model in Arc for sharing
        let model = std::sync::Arc::new(model);

        // 3. Create the chat node as a RunnableCallable
        let m = model.clone();
        let chat_node = RunnableCallable::new_sync("chat", move |input: &JsonValue, _config: &RunnableConfig| {
            // Extract messages from input
            let messages_json = match input.get("messages") {
                Some(JsonValue::Array(arr)) => arr.clone(),
                _ => return Err(RunnableError::Node("no messages in input".to_string())),
            };

            let messages: Vec<Message> = messages_json
                .iter()
                .filter_map(|j| serde_json::from_value(j.clone()).ok())
                .collect();

            // Call LLM
            let response = m.invoke(&messages, &RunnableConfig::new())
                .map_err(|e| RunnableError::Node(e.to_string()))?;

            let response_json = serde_json::to_value(&response)
                .map_err(|e| RunnableError::Node(e.to_string()))?;

            Ok(serde_json::json!({ "messages": [response_json] }))
        });

        // 4. Run single-turn
        println!("=== Single-turn Chat ===");
        println!("Model: {}", model_name);
        println!();

        let input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "你好！请用中文简单介绍一下你自己，不超过3句话。"}
            ]
        });

        println!("User: 你好！请用中文简单介绍一下你自己，不超过3句话。");
        println!();

        match chat_node.ainvoke(&input, &RunnableConfig::new()).await {
            Ok(output) => print_response(&output),
            Err(e) => eprintln!("Error: {}", e),
        }

        // 5. Run multi-turn
        println!();
        println!("=== Multi-turn Chat ===");
        println!();

        let multi_input = serde_json::json!({
            "messages": [
                {"type": "human", "content": "你好！请用中文简单介绍一下你自己，不超过3句话。"},
                {"type": "ai", "content": "你好！我是一个AI助手，可以帮你解答问题。有什么需要帮忙的吗？"},
                {"type": "human", "content": "1+1等于几？"}
            ]
        });

        println!("User: 1+1等于几？");
        println!();

        match chat_node.ainvoke(&multi_input, &RunnableConfig::new()).await {
            Ok(output) => print_response(&output),
            Err(e) => eprintln!("Error: {}", e),
        }

        Ok(())
    })
}

fn print_response(output: &JsonValue) {
    if let Some(messages) = output.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                println!("Assistant: {}", text);
            }
        }
    }
}
