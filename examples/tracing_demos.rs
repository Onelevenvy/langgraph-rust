use langgraph_tracing::{
    EventBus, InMemoryTracingStore, TraceStatus, TracingGraphObserver,
};
use serde_json::json;
use std::sync::Arc;

/// Demo: manually create traces and spans to test the tracing UI.
///
/// Run with:
///   cargo run --example tracing_demo
///
/// Then open http://127.0.0.1:3333 in your browser.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(InMemoryTracingStore::new());
    let event_bus = EventBus::new();

    // Start the web server in background
    let server_store = store.clone();
    let server_bus = event_bus.clone();
    tokio::spawn(async move {
        langgraph_tracing::server::start(
            "127.0.0.1:3333",
            server_store,
            server_bus,
            Some("crates/langgraph-tracing/frontend/dist"),
        )
        .await
        .unwrap();
    });

    // Give server a moment to start
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    println!("Server started at http://127.0.0.1:3333");
    println!("Creating demo traces...\n");

    // Create a few demo traces to populate the UI
    create_demo_traces(store.clone(), event_bus.clone()).await;

    println!("Demo traces created! Open http://127.0.0.1:3333 to view them.");
    println!("Press Ctrl+C to exit.");

    // Keep the server running
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn create_demo_traces(
    store: Arc<InMemoryTracingStore>,
    event_bus: EventBus,
) {
    // Trace 1: A successful ReAct agent run
    {
        let mut observer = TracingGraphObserver::new(store.clone(), event_bus.clone());
        let trace_id = observer.on_graph_start(
            "react_agent",
            json!({"messages": [{"type": "human", "content": "What's the weather in Tokyo?"}]}),
        );

        // LLM call
        let llm_span = observer.on_llm_start(&trace_id, None, "gpt-4o", json!([
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        observer.on_llm_end(
            &llm_span,
            &trace_id,
            json!({"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]}),
            true,
            Some(25),
            Some(30),
        );

        // Tool call
        let tool_span = observer.on_tool_start(
            &trace_id,
            Some(&llm_span),
            "get_weather",
            json!({"city": "Tokyo"}),
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        observer.on_tool_end(
            &tool_span,
            &trace_id,
            json!({"temperature": "22°C", "condition": "Partly cloudy", "humidity": "65%"}),
            true,
        );

        // Second LLM call with final answer
        let llm_span2 = observer.on_llm_start(&trace_id, None, "gpt-4o", json!([
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "args": {"city": "Tokyo"}}]},
            {"role": "tool", "content": "{\"temperature\": \"22°C\", \"condition\": \"Partly cloudy\"}"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(180)).await;
        observer.on_llm_end(
            &llm_span2,
            &trace_id,
            json!({"role": "assistant", "content": "The weather in Tokyo is currently 22°C and partly cloudy with 65% humidity."}),
            true,
            Some(80),
            Some(45),
        );

        observer.on_graph_end(
            &trace_id,
            json!({"messages": [{"type": "ai", "content": "The weather in Tokyo is currently 22°C and partly cloudy."}]}),
            TraceStatus::Success,
        );
    }

    // Trace 2: A failed graph run
    {
        let mut observer = TracingGraphObserver::new(store.clone(), event_bus.clone());
        let trace_id = observer.on_graph_start(
            "data_pipeline",
            json!({"input": "process dataset"}),
        );

        let node_span = observer.on_node_start(&trace_id, None, "fetch_data", json!({"source": "api"}));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        observer.on_node_end(&node_span, &trace_id, json!({"rows": 1000}), true);

        let node_span2 = observer.on_node_start(&trace_id, None, "transform", json!({"operation": "normalize"}));
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        observer.on_node_end(&node_span2, &trace_id, json!({"error": "Column 'value' not found"}), false);

        observer.on_graph_end(&trace_id, json!({"error": "Transform failed"}), TraceStatus::Error);
    }

    // Trace 3: A simple successful run
    {
        let mut observer = TracingGraphObserver::new(store.clone(), event_bus.clone());
        let trace_id = observer.on_graph_start(
            "chat_agent",
            json!({"messages": [{"type": "human", "content": "Hello!"}]}),
        );

        let llm_span = observer.on_llm_start(&trace_id, None, "gpt-4o-mini", json!([
            {"role": "user", "content": "Hello!"}
        ]));
        tokio::time::sleep(std::time::Duration::from_millis(90)).await;
        observer.on_llm_end(
            &llm_span,
            &trace_id,
            json!({"role": "assistant", "content": "Hi there! How can I help you today?"}),
            true,
            Some(12),
            Some(10),
        );

        observer.on_graph_end(
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
