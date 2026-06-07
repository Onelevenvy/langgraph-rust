/// parallel_interrupt_hitl.rs
///
/// The scenario: in a single super-step, worker_a completes (writing a
/// `branch:to:output` trigger channel) while worker_b is interrupted.
/// The fix ensures worker_a's channel write survives into the checkpoint.
/// After resume, the output node is triggered correctly.
///
/// Graph:
///   START → entry
///   entry → worker_a   (completes, writes branch:to:output)
///   entry → worker_b   (calls interrupt() to request human review)
///   worker_a → output  (downstream of worker_a only)
///   worker_b → END
///   output   → END
///

use std::sync::Arc;


use serde_json::{json, Value as JsonValue};

use langgraph::prelude::*;
use langgraph::checkpoint::InMemorySaver;
use langgraph::langgraph_state;

// ── State ─────────────────────────────────────────────────────────────────

#[langgraph_state]
#[derive(Debug)]
struct PipelineState {
    #[channel(messages)]
    log: Vec<String>,
    review_value: String,
    output_result: String,
}

// ── Nodes ──────────────────────────────────────────────────────────────────

async fn entry_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[entry] running");
    Ok(json!({ "log": ["entry ran"] }))
}

/// Worker A: completes immediately in the same super-step as worker_b.
/// Its branch:to:output trigger write must survive the interrupt checkpoint.
async fn worker_a_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[worker_a] completed → writing state + downstream trigger");
    Ok(json!({ "log": ["worker_a done"] }))
}

/// Worker B: interrupts in the same super-step as worker_a.
async fn worker_b_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[worker_b] calling interrupt() ...");
    let review = interrupt(json!({
        "question": "Please review and provide input."
    }))
    .map_err(|e| RunnableError::Interrupt(e.into()))?;

    let review_str = if let Some(s) = review.as_str() {
        s.to_string()
    } else {
        review.to_string()
    };
    println!("[worker_b] resumed with: {:?}", review_str);
    Ok(json!({
        "log": [format!("worker_b done, review={}", review_str)],
        "review_value": review_str,
    }))
}

/// Output node: triggered only by worker_a's edge.

async fn output_node(_input: JsonValue, _config: RunnableConfig) -> Result<JsonValue, RunnableError> {
    println!("[output] running! ");
    Ok(json!({
        "log": ["output ran"],
        "output_result": "output complete",
    }))
}

// ── Main ───────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Parallel Interrupt HITL");
    println!("========================================");
    println!();
    println!("  Graph:");
    println!("    START → entry → worker_a → output → END");
    println!("                 → worker_b → END");
    println!();
    println!("  worker_a and worker_b run in the SAME super-step.");
    println!("  worker_a completes (writes branch:to:output trigger).");
    println!("  worker_b calls interrupt().");
    println!();
    println!("  WITHOUT fix: branch:to:output is lost → output never runs.");
    println!("  WITH fix:    branch:to:output survives → output runs after resume.");
    println!();

    let channels = PipelineState::create_channels();
    let mut graph = StateGraph::new(channels);

    graph.add_node("entry",    entry_node)?;
    graph.add_node("worker_a", worker_a_node)?;
    graph.add_node("worker_b", worker_b_node)?;
    graph.add_node("output",   output_node)?;

    graph.add_edge(START, "entry")?;
    // entry fans out — both workers triggered in the SAME super-step
    graph.add_edge("entry", "worker_a")?;
    graph.add_edge("entry", "worker_b")?;
   
    graph.add_edge("worker_a", "output")?;
    graph.add_edge("worker_b", END)?;
    graph.add_edge("output", END)?;

    let checkpointer = Arc::new(InMemorySaver::new());
    let compiled = graph
        .compile_builder()
        .checkpointer(checkpointer)
        .build()?;

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        json!({ "thread_id": "thread-1" }),
    );

    // ── Step 1: initial invoke (graph pauses at worker_b interrupt) ──────

    println!("--- Step 1: Initial invoke ---");
    println!();

    let input = json!({ "log": [], "review_value": "", "output_result": "" });
    let _result1 = compiled.ainvoke(&input, &config).await?;
    println!();

    // ── Inspect snapshot ─────────────────────────────────────────────────

    let snapshot = compiled.get_state(&config)?;
    println!("snapshot.next = {:?}", snapshot.next);
    println!();

    // Graph must be paused (worker_b interrupted, possibly worker_a also pending)
    assert!(
        !snapshot.next.is_empty(),
        "Graph should be paused, but next is empty: {:?}", snapshot.next
    );

    // Depending on task order:
    // Case A: worker_a ran first, then worker_b interrupted
    //   → worker_a's log is in checkpoint, output trigger is in checkpoint
    //   → snapshot.next contains worker_b (+ maybe output)
    // Case B: worker_b interrupted first (before worker_a ran)
    //   → worker_a is still in snapshot.next
    //   → after resume: worker_a runs, then output runs
    //
    // In BOTH cases, after resume output must run.

    let log_now = snapshot.values["log"].as_array().cloned().unwrap_or_default();
    if log_now.iter().any(|v| v.as_str().map(|s| s.contains("worker_a")).unwrap_or(false)) {
        println!("ℹ worker_a completed in the same super-step as worker_b's interrupt.");
        println!("  This is the direct Bug  scenario.");
        println!("  Verifying worker_a's writes survived the interrupt checkpoint...");
        // The key check: branch:to:output must exist in the checkpoint so
        // output runs after resume. We verify this indirectly by checking
        // the final output below.
    } else {
        println!("ℹ worker_b interrupted before worker_a ran.");
        println!("  After resume, worker_a will run, then output.");
    }
    println!();

    // ── Step 2: resume ───────────────────────────────────────────────────

    println!("--- Step 2: Resume with human input ---");
    println!("H: \"Approved\"");
    println!();

    let resume_cmd = Command::resume(json!("Approved"));
    let result2 = compiled
        .ainvoke(&serde_json::to_value(&resume_cmd)?, &config)
        .await?;

    println!();
    println!("--- Final state ---");
    print_state(&result2);
    println!();

    // ── Assertions ───────────────────────────────────────────────────────

    let final_log = result2["log"].as_array().cloned().unwrap_or_default();
    println!("Final log: {:?}", final_log);
    println!();

    let has_entry    = final_log.iter().any(|v| v.as_str().map(|s| s.contains("entry")).unwrap_or(false));
    let has_worker_a = final_log.iter().any(|v| v.as_str().map(|s| s.contains("worker_a")).unwrap_or(false));
    let has_worker_b = final_log.iter().any(|v| v.as_str().map(|s| s.contains("worker_b")).unwrap_or(false));
    let has_output   = final_log.iter().any(|v| v.as_str().map(|s| s == "output ran").unwrap_or(false));

    println!("Assertions:");
    assert!(has_entry, "FAIL: entry missing from final log");
    println!("  ✅ entry ran");
    assert!(has_worker_a, "FAIL: worker_a missing from final log");
    println!("  ✅ worker_a ran");
    assert!(has_worker_b, "FAIL: worker_b missing from final log");
    println!("  ✅ worker_b ran");
    assert!(
        has_output,
        "FAIL: output node never ran!\n\
         Bug regression: worker_a's branch:to:output trigger was lost\n\
         when worker_b interrupted. The fix ensures completed tasks' writes\n\
         are applied to channels before saving the interrupt checkpoint."
    );
    println!("  ✅ output ran  ← Bug  regression check passed!");

    let output_result = result2["output_result"].as_str().unwrap_or("");
    assert_eq!(output_result, "output complete");
    println!("  ✅ output_result = {:?}", output_result);

    let review = result2["review_value"].as_str().unwrap_or("");
    assert_eq!(review, "Approved");
    println!("  ✅ review_value = {:?}", review);

    println!();
    println!("========================================");
    println!("  All assertions passed!");
    println!("========================================");
    Ok(())
}

fn print_state(state: &JsonValue) {
    if let Some(obj) = state.as_object() {
        for (k, v) in obj {
            if k.starts_with("branch:") || k.starts_with("join:") || k == "__start__" {
                continue;
            }
            println!("  {k}: {v}");
        }
    }
}
