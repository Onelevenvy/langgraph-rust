
use dotenvy::dotenv;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_derive::tool;
use langgraph_prebuilt::{create_react_agent, prepare_tools, ReActAgentConfig};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};

use std::sync::Arc;

fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

// -------------------------------------------------------
// 1. 使用 #[tool] 宏定义工具
// -------------------------------------------------------

/// 获取指定城市的当前天气。
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

/// 计算简单的数学表达式。
#[tool(
    "calculator",
    "Evaluate a mathematical expression. Supports basic arithmetic: +, -, *, /"
)]
fn calculator(expression: String) -> Result<String, String> {
    let result = match expression.replace(" ", "").as_str() {
        "2+2" => "4",
        "10*5" => "50",
        "100/4" => "25",
        _ => return Ok(format!("Cannot evaluate: {}", expression)),
    };
    Ok(format!("{} = {}", expression, result))
}

/// 查询内部知识库关于 Rust、LangGraph。
#[tool(
    "search_knowledge",
    "Search the internal knowledge base for information about a topic."
)]
fn search_knowledge(query: String) -> Result<String, String> {
    let answer = match query.to_lowercase().as_str() {
        q if q.contains("rust") => {
            "Rust is a systems programming language focused on safety, speed, and concurrency."
        }
        q if q.contains("langgraph") => {
            "LangGraph is a library for building stateful, multi-actor applications with LLMs."
        }
        _ => "No relevant information found in the knowledge base.",
    };
    Ok(answer.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  LangGraph ReAct Agent (Macro Tools)");
    println!("========================================\n");

    // -------------------------------------------------------
    // 2. 准备工具
    // -------------------------------------------------------

    let prepared = prepare_tools(vec![
        Arc::new(GetWeather::new()),
        Arc::new(Calculator::new()),
        Arc::new(SearchKnowledge::new()),
    ]);

    // -------------------------------------------------------
    // 3. 创建 LLM 模型
    // -------------------------------------------------------

    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.7),
        ..Default::default()
    });

    // -------------------------------------------------------
    // 4. 创建 ReAct Agent
    // -------------------------------------------------------

    // 注意：这里传入的是 prepared.tools (符合 Vec<Arc<dyn BaseTool>> 类型)
    let agent = create_react_agent(
        Arc::new(model),
        prepared.tools,
        Some(ReActAgentConfig {
            system_prompt: Some(
                "You are a helpful assistant with access to tools. \
                 Use the tools when needed to answer user questions."
                    .to_string(),
            ),
            max_steps: Some(10),
            handle_tool_errors: true,
        }),
    )?;

    // -------------------------------------------------------
    // 5. 测试运行
    // -------------------------------------------------------

    let queries = vec![
        "What's the weather like in Beijing?",
        "What is 100 divided by 4?",
        "Tell me about the Rust programming language.",
        "What's the weather in Shanghai and what is 2+2?",
    ];

    for query in queries {
        println!("--- Query: {} ---", query);
        let input = serde_json::json!({
            "messages": [{"type": "human", "content": query}]
        });

        let result = agent.ainvoke(&input, &RunnableConfig::new()).await?;
        print_agent_response(&result);
        println!();
    }

    Ok(())
}

fn print_agent_response(output: &serde_json::Value) {
    if let Some(messages) = output.get("messages").and_then(|m| m.as_array()) {
        for msg in messages.iter().rev() {
            if let Some(msg_type) = msg.get("type").and_then(|t| t.as_str()) {
                if msg_type == "ai" {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            println!("Assistant: {}", content);
                            return;
                        }
                    }
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        if !tool_calls.is_empty() {
                            println!("Assistant: [Called {} tool(s)]", tool_calls.len());
                            return;
                        }
                    }
                }
            }
        }
    }
}
