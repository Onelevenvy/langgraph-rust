use langgraph_derive::Traceable;
use langgraph_tracing::{TraceStatus, TracingContext};
use serde_json::json;

/// Demo: manually create traces and spans to test the tracing UI.
///
/// Run with:
///   cargo run --example tracing_demo
///
/// Then open http://127.0.0.1:3333 in your browser.
#[derive(Traceable)]
struct DemoState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ctx = DemoState::tracing_context();
    ctx.start_server("127.0.0.1:3333");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    println!("Server started at http://127.0.0.1:3333");
    println!("Creating demo traces...\n");

    create_demo_traces(&mut ctx).await;

    println!("Demo traces created! Open http://127.0.0.1:3333 to view them.");
    println!("Press Ctrl+C to exit.");
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn create_demo_traces(ctx: &mut TracingContext) {
    // Trace 1: A successful ReAct agent run
    {
        let obs = ctx.observer();
        let trace_id = obs.on_graph_start(
            "react_agent",
            json!({"messages": [{"type": "human", "content": "What's the weather in Tokyo?"}]}),
        );

        let llm_span = obs.on_llm_start(&trace_id, None, "gpt-4o", json!([
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        obs.on_llm_end(
            &llm_span,
            &trace_id,
            json!({"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]}),
            true,
            Some(25),
            Some(30),
        );

        let tool_span = obs.on_tool_start(
            &trace_id,
            Some(&llm_span),
            "get_weather",
            json!({"city": "Tokyo"}),
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        obs.on_tool_end(
            &tool_span,
            &trace_id,
            json!({"temperature": "22°C", "condition": "Partly cloudy", "humidity": "65%"}),
            true,
        );

        let llm_span2 = obs.on_llm_start(&trace_id, None, "gpt-4o", json!([
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]},
            {"role": "tool", "content": "{\"temperature\": \"22°C\", \"condition\": \"Partly cloudy\"}"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(180)).await;
        obs.on_llm_end(
            &llm_span2,
            &trace_id,
            json!({"role": "assistant", "content": "The weather in Tokyo is currently 22°C and partly cloudy with 65% humidity."}),
            true,
            Some(80),
            Some(45),
        );

        obs.on_graph_end(
            &trace_id,
            json!({"messages": [{"type": "ai", "content": "The weather in Tokyo is currently 22°C and partly cloudy."}]}),
            TraceStatus::Success,
        );
    }

    // Trace 2: A failed graph run
    {
        let obs = ctx.observer();
        let trace_id = obs.on_graph_start(
            "data_pipeline",
            json!({"input": "process dataset"}),
        );

        let node_span = obs.on_node_start(&trace_id, None, "fetch_data", json!({"source": "api"}));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        obs.on_node_end(&node_span, &trace_id, json!({"rows": 1000}), true);

        let node_span2 = obs.on_node_start(&trace_id, None, "transform", json!({"operation": "normalize"}));
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        obs.on_node_end(&node_span2, &trace_id, json!({"error": "Column 'value' not found"}), false);

        obs.on_graph_end(&trace_id, json!({"error": "Transform failed"}), TraceStatus::Error);
    }

    // Trace 3: A simple successful run
    {
        let obs = ctx.observer();
        let trace_id = obs.on_graph_start(
            "chat_agent",
            json!({"messages": [{"type": "human", "content": "Hello!"}]}),
        );

        let llm_span = obs.on_llm_start(&trace_id, None, "gpt-4o-mini", json!([
            {"role": "user", "content": "Hello!"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(90)).await;
        obs.on_llm_end(
            &llm_span,
            &trace_id,
            json!({"role": "assistant", "content": "Hi there! How can I help you today?"}),
            true,
            Some(12),
            Some(10),
        );

        obs.on_graph_end(
            &trace_id,
            json!({"messages": [{"type": "ai", "content": "Hi there! How can I help you today?"}]}),
            TraceStatus::Success,
        );
    }

    println!("  Created 3 demo traces:");
    println!("  1. react_agent  (success, 3 spans, with LLM + tool calls)");
    println!("  2. data_pipeline (error, 2 spans, failed at transform)");
    println!("  3. chat_agent    (success, 1 span, simple LLM call)");
}
