//! Benchmark: message-history accumulation through the channel reducer.
//!
//! This is the per-agent-turn hot path: a growing `messages` array is merged by
//! `add_messages_ref` on every LLM step. Run explicitly (it is `#[ignore]`d so
//! normal `cargo test` runs stay fast):
//!
//! ```text
//! cargo test --release -p langgraph-prebuilt --test bench_messages -- --ignored --nocapture
//! ```

use langgraph::channels::{BinaryOperatorAggregate, Channel};
use langgraph_prebuilt::add_messages_ref;
use serde_json::Value as JsonValue;
use std::time::{Duration, Instant};

fn make_message(i: usize) -> JsonValue {
    // A realistic AI message with tool call args, sized so deep clones cost real time.
    serde_json::json!({
        "type": "ai",
        "content": format!("Assistant reply number {i}: {}", "x".repeat(300)),
        "tool_calls": [{
            "name": "search_tool",
            "args": {
                "query": format!("query-{i}"),
                "filters": {"category": "test", "limit": 10, "extra": "data"}
            },
            "id": format!("call_{i}")
        }],
        "id": format!("msg_{i}")
    })
}

/// Time appending `steps` messages to a growing history, best of `iterations`.
fn run_case(steps: usize, iterations: u32) -> Duration {
    let messages: Vec<JsonValue> = (0..steps).map(make_message).collect();

    let mut best = Duration::MAX;
    for _ in 0..iterations {
        let ch = BinaryOperatorAggregate::new("messages", add_messages_ref);
        // Seed with an empty array, exactly like a real graph initializes the
        // messages channel, so each update goes through the reducer as an append.
        ch.update(&[serde_json::json!([])]).unwrap();
        let start = Instant::now();
        for msg in &messages {
            ch.update(std::slice::from_ref(msg)).unwrap();
        }
        best = best.min(start.elapsed());
    }
    best
}

#[test]
#[ignore]
fn bench_message_accumulation() {
    for steps in [250usize, 500, 1000] {
        let elapsed = run_case(steps, 3);
        println!("add_messages: {steps:>4} steps (with tool calls) => best {elapsed:?}");
    }

    // Sanity: the channel really accumulates one message per update.
    let ch = BinaryOperatorAggregate::new("messages", add_messages_ref);
    ch.update(&[serde_json::json!([])]).unwrap();
    ch.update(&[make_message(0)]).unwrap();
    ch.update(&[make_message(1)]).unwrap();
    assert_eq!(ch.get().unwrap().as_array().unwrap().len(), 2);
}
