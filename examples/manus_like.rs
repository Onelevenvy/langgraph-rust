use std::sync::Arc;

use serde_json::Value as JsonValue;

use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::langgraph_state;
use langgraph::prebuilt::{ask_json, print_stream, response_text, stream_llm, BaseChatModel, Message};
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
// State
// -------------------------------------------------------

#[langgraph_state]
#[derive(Debug)]
struct ManusState {
    #[channel(messages)]
    messages: Vec<Message>,

    #[channel]
    plan: JsonValue,
}

// -------------------------------------------------------
// Helpers
// -------------------------------------------------------

/// Count pending steps in a plan.
fn pending_steps(plan: &JsonValue) -> usize {
    plan.get("steps")
        .and_then(|s| s.as_array())
        .map(|steps| {
            steps
                .iter()
                .filter(|s| s.get("status").and_then(|v| v.as_str()) != Some("completed"))
                .count()
        })
        .unwrap_or(0)
}

/// Get last user message from input.
fn user_message(input: &JsonValue) -> &str {
    input
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
}

// -------------------------------------------------------
// Prompts
// -------------------------------------------------------

const CREATE_PLAN_PROMPT: &str = r#"You are a planning agent. Given the user's request, create a step-by-step plan.

Respond in JSON format ONLY (no markdown fences):
{
  "title": "Plan title",
  "steps": [
    { "description": "Step 1 description", "status": "pending" },
    { "description": "Step 2 description", "status": "pending" }
  ]
}

Keep the plan concise with 2-5 actionable steps."#;

const EXECUTE_STEP_PROMPT: &str = r#"You are an execution agent. Execute the current step of the plan.

Respond in JSON format ONLY (no markdown fences):
{
  "success": true,
  "result": "Description of what was done and the result"
}

If the step fails, set success to false and explain the error in result."#;

const REPLAN_PROMPT: &str = r#"You are a planning agent. A step has just been completed. Review and update the plan.

Respond with the FULL updated plan in JSON format ONLY (no markdown fences):
{
  "title": "Plan title",
  "steps": [
    { "description": "Step description", "status": "completed", "result": "what happened" },
    { "description": "Next step description", "status": "pending" }
  ]
}

Mark completed steps with status "completed". Adjust remaining steps based on what was learned."#;

const SUMMARIZE_PROMPT: &str = r#"All plan steps have been completed. Write a concise summary of what was accomplished.

Respond in JSON format ONLY (no markdown fences):
{
  "message": "Summary of the work done"
}"#;

// -------------------------------------------------------
// Build graph
// -------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Manus-like Plan-and-Act Agent");
    println!("========================================\n");

    let (api_key, api_base, model_name) = load_openai_config();
    let model: Arc<dyn BaseChatModel> = Arc::new(OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        temperature: Some(0.3),
        ..Default::default()
    }));

    let channels = ManusState::create_channels();
    let mut graph = StateGraph::new(channels);

    // --- Node: planner (create plan) ---
    let m = model.clone();
    graph.add_node("planner", move |input: JsonValue, _config: RunnableConfig| {
        let model = m.clone();
        async move {
            let prompt = format!("{}\n\nUser request: {}", CREATE_PLAN_PROMPT, user_message(&input));
            let plan = ask_json(model.as_ref(), &prompt, "").await?
                .unwrap_or_else(|| serde_json::json!({"title": "Untitled", "steps": []}));

            let step_count = plan.get("steps").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0);
            println!("[planner] Created plan: {} ({} steps)",
                plan.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled"), step_count);
            for (i, step) in plan.get("steps").and_then(|s| s.as_array()).unwrap_or(&vec![]).iter().enumerate() {
                println!("  {}. {}", i + 1, step.get("description").and_then(|d| d.as_str()).unwrap_or(""));
            }

            Ok(serde_json::json!({"plan": plan}))
        }
    })?;

    // --- Node: executor (execute one step) ---
    let m = model.clone();
    graph.add_node("executor", move |input: JsonValue, _config: RunnableConfig| {
        let model = m.clone();
        async move {
            let plan = input.get("plan").cloned().unwrap_or_default();
            let steps = plan.get("steps").and_then(|s| s.as_array()).cloned().unwrap_or_default();

            let idx = match steps.iter().position(|s| s.get("status").and_then(|v| v.as_str()) != Some("completed")) {
                Some(i) => i,
                None => return Ok(serde_json::json!({"plan": plan})),
            };

            let desc = steps[idx].get("description").and_then(|d| d.as_str()).unwrap_or("?");
            println!("\n[executor] Step {}/{}: {}", idx + 1, steps.len(), desc);

            let prompt = format!("{}\n\nCurrent step: {}", EXECUTE_STEP_PROMPT, desc);
            let exec = ask_json(model.as_ref(), &prompt, "").await?
                .unwrap_or_else(|| serde_json::json!({"success": false, "result": "Parse error"}));

            let success = exec.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
            let step_result = exec.get("result").and_then(|r| r.as_str()).unwrap_or("");
            if success { println!("[executor] Done: {}", step_result); }
            else { println!("[executor] Failed: {}", step_result); }

            // Mark step completed
            let mut updated_plan = plan;
            if let Some(steps) = updated_plan.get_mut("steps").and_then(|s| s.as_array_mut()) {
                if let Some(step) = steps.get_mut(idx) {
                    if let Some(obj) = step.as_object_mut() {
                        obj.insert("status".to_string(), serde_json::json!("completed"));
                        obj.insert("result".to_string(), serde_json::json!(step_result));
                    }
                }
            }

            Ok(serde_json::json!({"plan": updated_plan}))
        }
    })?;

    // --- Node: replanner (update plan after step) ---
    let m = model.clone();
    graph.add_node("replanner", move |input: JsonValue, _config: RunnableConfig| {
        let model = m.clone();
        async move {
            let plan = input.get("plan").cloned().unwrap_or_default();
            println!("[replanner] {} steps remaining, updating plan...", pending_steps(&plan));

            let prompt = format!("{}\n\nCurrent plan:\n{}", REPLAN_PROMPT, serde_json::to_string_pretty(&plan).unwrap_or_default());
            let updated_plan = ask_json(model.as_ref(), &prompt, "").await?
                .unwrap_or(plan);

            println!("[replanner] Plan updated, {} steps remaining", pending_steps(&updated_plan));
            Ok(serde_json::json!({"plan": updated_plan}))
        }
    })?;

    // --- Node: summarizer ---
    let m = model.clone();
    graph.add_node("summarizer", move |input: JsonValue, _config: RunnableConfig| {
        let model = m.clone();
        async move {
            let plan = input.get("plan").cloned().unwrap_or_default();
            println!("[summarizer] Generating summary...");

            let prompt = format!("{}\n\nCompleted plan:\n{}", SUMMARIZE_PROMPT, serde_json::to_string_pretty(&plan).unwrap_or_default());
            let result = stream_llm(model.as_ref(), &serde_json::json!({"messages": [{"type": "human", "content": prompt}]}), "").await?;
            let text = response_text(&result);

            println!("[summarizer] Done\n=== Summary ===\n{}\n===============", text);
            Ok(serde_json::json!({"messages": [{"type": "ai", "content": text}]}))
        }
    })?;

    // --- Edges ---
    graph.add_edge(START, "planner")?;
    graph.add_edge("planner", "executor")?;
    conditional_edges!(graph, "executor", route_after_executor, "replanner" => "replanner", END => END)?;
    conditional_edges!(graph, "replanner", route_after_replanner, "executor" => "executor", "summarizer" => "summarizer", END => END)?;
    graph.add_edge("summarizer", END)?;

    let app = graph.compile()?;

    // -------------------------------------------------------
    // Run
    // -------------------------------------------------------

    let input = serde_json::json!({
        "messages": [{ "type": "human", "content": "规划深圳的3天旅游." }]
    });

    println!("User: 规划深圳的3天旅游.\n");
    let mut stream = app.astream(&input, &RunnableConfig::new(), vec![StreamMode::Custom, StreamMode::Updates]);

    // Use print_stream helper — replaces ~15 lines of manual token printing
    let _ = print_stream(&mut stream, true).await;

    println!("\n========================================\n  Demo completed!\n========================================");
    Ok(())
}

// -------------------------------------------------------
// Routing
// -------------------------------------------------------

fn route_after_executor(input: &JsonValue) -> String {
    let pending = pending_steps(&input.get("plan").cloned().unwrap_or_default());
    if pending == 0 { return END.to_string(); }
    println!("[route] {} steps remaining → replanner", pending);
    "replanner".to_string()
}

fn route_after_replanner(input: &JsonValue) -> String {
    let pending = pending_steps(&input.get("plan").cloned().unwrap_or_default());
    if pending == 0 { return "summarizer".to_string(); }
    println!("[route] {} steps remaining → executor", pending);
    "executor".to_string()
}
