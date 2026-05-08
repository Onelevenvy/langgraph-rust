/// join_edge_test.rs
///
/// Tests `add_join_edge`: a fan-in node that waits for ALL upstream nodes
/// to complete before it runs.
///
/// Graph:
///   START → entry → worker_a ─┐
///                  → worker_b ─┴→ merger (join) → output → END
///

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use langgraph::prelude::*;
use langgraph_checkpoint::checkpoint::memory::InMemorySaver;
use langgraph_derive::StateGraph;

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default, StateGraph)]
struct FanInState {
    #[channel(messages)]
    log: Vec<String>,
    output: String,
}

// ── Nodes ──────────────────────────────────────────────────────────────────

async fn entry_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[entry] running");
    Ok(json!({ "log": ["entry ran"] }))
}

async fn worker_a_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[worker_a] completed");
    Ok(json!({ "log": ["worker_a done"] }))
}

async fn worker_b_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[worker_b] completed");
    Ok(json!({ "log": ["worker_b done"] }))
}

/// Merger: should ONLY run after BOTH worker_a and worker_b complete.
async fn merger_node(input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    let log_len = input["log"].as_array().map(|a| a.len()).unwrap_or(0);
    println!("[merger] running with {} log entries", log_len);
    // Verify both workers contributed to the log before merger ran
    let log = input["log"].as_array().cloned().unwrap_or_default();
    let has_a = log.iter().any(|v| v.as_str().map(|s| s.contains("worker_a")).unwrap_or(false));
    let has_b = log.iter().any(|v| v.as_str().map(|s| s.contains("worker_b")).unwrap_or(false));
    if has_a && has_b {
        println!("[merger] both workers' results are visible ✓");
    } else {
        eprintln!("[merger] WARNING: only some workers visible — join may not be working correctly");
    }
    Ok(json!({ "log": ["merger ran"] }))
}

async fn output_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[output] running");
    Ok(json!({
        "log": ["output ran"],
        "output": "pipeline complete",
    }))
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Join Edge Test");
    println!("========================================");
    println!();
    println!("  START → entry → worker_a ─┐");
    println!("                 → worker_b ─┴→ merger (join) → output → END");
    println!();
    println!("  merger must run AFTER both worker_a and worker_b complete.");
    println!();

    let channels = FanInState::create_channels();
    let mut graph = StateGraph::new(channels);

    graph.add_node("entry",   entry_node)?;
    graph.add_node("worker_a", worker_a_node)?;
    graph.add_node("worker_b", worker_b_node)?;
    graph.add_node("merger",  merger_node)?;
    graph.add_node("output",  output_node)?;

    graph.add_edge(START, "entry")?;
    graph.add_edge("entry", "worker_a")?;
    graph.add_edge("entry", "worker_b")?;
    // join: merger waits for BOTH workers
    graph.add_join_edge(vec!["worker_a".into(), "worker_b".into()], "merger")?;
    graph.add_edge("merger", "output")?;
    graph.add_edge("output", END)?;

    let compiled = graph.compile()?;

    println!("--- Running graph ---");
    println!();

    let input = json!({ "log": [], "output": "" });
    let result = compiled.ainvoke(&input, &RunnableConfig::new()).await?;

    println!();
    println!("--- Final state ---");
    if let Some(obj) = result.as_object() {
        for (k, v) in obj {
            if k.starts_with("branch:") || k.starts_with("join:") || k == "__start__" {
                continue;
            }
            println!("  {k}: {v}");
        }
    }
    println!();

    let final_log = result["log"].as_array().cloned().unwrap_or_default();
    println!("Final log: {:?}", final_log);
    println!();

    let has_entry    = final_log.iter().any(|v| v.as_str().map(|s| s.contains("entry")).unwrap_or(false));
    let has_worker_a = final_log.iter().any(|v| v.as_str().map(|s| s.contains("worker_a")).unwrap_or(false));
    let has_worker_b = final_log.iter().any(|v| v.as_str().map(|s| s.contains("worker_b")).unwrap_or(false));
    let has_merger   = final_log.iter().any(|v| v.as_str().map(|s| s.contains("merger")).unwrap_or(false));
    let has_output   = final_log.iter().any(|v| v.as_str().map(|s| s.contains("output")).unwrap_or(false));

    println!("Assertions:");
    assert!(has_entry,    "FAIL: entry missing");    println!("  ✅ entry ran");
    assert!(has_worker_a, "FAIL: worker_a missing"); println!("  ✅ worker_a ran");
    assert!(has_worker_b, "FAIL: worker_b missing"); println!("  ✅ worker_b ran");
    assert!(
        has_merger,
        "FAIL: merger never ran!\n\
         The join barrier was not triggered after both workers completed."
    );
    println!("  ✅ merger ran (join barrier worked!)");
    assert!(has_output,   "FAIL: output missing");   println!("  ✅ output ran");

    let out = result["output"].as_str().unwrap_or("");
    assert_eq!(out, "pipeline complete");
    println!("  ✅ output = {:?}", out);

    // Verify merger ran AFTER both workers (log ordering)
    let positions: Vec<usize> = ["entry ran", "worker_a done", "worker_b done", "merger ran", "output ran"]
        .iter()
        .filter_map(|needle| {
            final_log.iter().position(|v| v.as_str().map(|s| s == *needle).unwrap_or(false))
        })
        .collect();

    if positions.len() == 5 {
        let merger_pos = positions[3];
        let worker_a_pos = positions[1];
        let worker_b_pos = positions[2];
        assert!(
            merger_pos > worker_a_pos && merger_pos > worker_b_pos,
            "FAIL: merger ran before workers completed! positions={:?}", positions
        );
        println!("  ✅ merger ran after both workers (ordering correct)");
    }

    println!();
    println!("========================================");
    println!("  All assertions passed!");
    println!("========================================");
    Ok(())
}
