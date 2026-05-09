//! Example: Call a reasoning model and display thinking process.
//!
//! OpenAIModel now natively supports `reasoning_content` from reasoning models
//! (DeepSeek, SiliconFlow, o1/o3, etc.). If the model returns reasoning_content,
//! it will be displayed; otherwise only the answer is shown.
//!
//! # Environment Variables
//! - OPENAI_API_KEY (or DEEPSEEK_API_KEY)
//! - OPENAI_API_BASE (optional, defaults to https://api.openai.com/v1)
//! - OPENAI_MODEL (optional, defaults to deepseek-reasoner)
//!
//! # Run
//! cargo run --example thinking_model

use dotenvy::dotenv;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_prebuilt::{BaseChatModel, Message};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        dotenv().ok();

        let api_key = std::env::var("OPENAI_API_KEY")
            .expect("Set OPENAI_API_KEY or DEEPSEEK_API_KEY");
        let api_base = std::env::var("OPENAI_API_BASE")
            .ok();
        let model_name = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "deepseek-reasoner".to_string());

        let model = OpenAIModel::new(OpenAIModelConfig {
            model: model_name.clone(),
            api_key,
            api_base,
            ..Default::default()
        });

        println!("=== Thinking Model Demo ===");
        println!("Model: {}", model_name);
        println!();

        // --- Non-streaming ---
        println!("--- Non-streaming ---");
        println!("User: 请用中文解释什么是快速排序算法，时间复杂度是多少？");
        println!();

        let messages = vec![Message::human(
            "请用中文解释什么是快速排序算法，时间复杂度是多少？",
        )];

        let response = model.ainvoke(&messages, &RunnableConfig::new()).await?;

        if let Some(thinking) = response.thinking() {
            println!("[Thinking]");
            println!("{}", thinking);
            println!();
        }
        println!("[Answer]");
        println!("{}", response.text().unwrap_or("(empty)"));
        println!();

        // --- Streaming ---
        println!("--- Streaming ---");
        println!("User: 用中文简要说明斐波那契数列的定义和前10项。");
        println!();

        let messages2 = vec![Message::human(
            "用中文简要说明斐波那契数列的定义和前10项。",
        )];

        let config = RunnableConfig::new();
        let mut stream = model.astream(&messages2, &config);
        use tokio_stream::StreamExt;
        let mut in_thinking = false;

        while let Some(result) = stream.next().await {
            let msg = result?;
            if let Some(t) = msg.thinking() {
                if !t.is_empty() && !in_thinking {
                    print!("[Thinking] ");
                    in_thinking = true;
                }
                if !t.is_empty() {
                    print!("{}", t);
                }
            }
            if let Some(text) = msg.text() {
                if !text.is_empty() {
                    if in_thinking {
                        println!();
                        in_thinking = false;
                        print!("[Answer] ");
                    }
                    print!("{}", text);
                }
            }
        }
        println!();

        Ok(())
    })
}
