//! Minimal demonstration of `SqliteSaver`: put a checkpoint, fetch it
//! back, list history, store pending writes, then delete the thread.
//!
//! Run with:
//!     cargo run --example sqlite_checkpoint

use std::collections::HashMap;

use langgraph::checkpoint::BaseCheckpointSaver;
use langgraph::checkpoint::checkpoint::types::{
    ChannelVersions, Checkpoint, CheckpointMetadata, CheckpointSource,
};
use langgraph::prelude::RunnableConfig;
use langgraph::sqlite::SqliteSaver;
use serde_json::Value as JsonValue;

fn config_for(thread_id: &str) -> RunnableConfig {
    serde_json::from_value(serde_json::json!({
        "configurable": { "thread_id": thread_id, "checkpoint_ns": "" }
    }))
    .unwrap()
}

/// Build a checkpoint with explicit per-channel versions. Blob storage
/// is keyed by (channel, version) and immutable on write, so callers
/// MUST increment versions whenever a channel value changes.
fn make_checkpoint(channel_values: Vec<(&str, JsonValue, i64)>) -> (Checkpoint, ChannelVersions) {
    let mut cp = Checkpoint::empty();
    let mut versions: ChannelVersions = HashMap::new();
    for (k, v, ver) in channel_values {
        cp.channel_values.insert(k.to_string(), v);
        cp.channel_versions
            .insert(k.to_string(), JsonValue::Number(ver.into()));
        versions.insert(k.to_string(), JsonValue::Number(ver.into()));
    }
    (cp, versions)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use an in-memory database for a self-contained demo. Swap for
    // "sqlite:./checkpoints.db" to persist across runs.
    let saver = SqliteSaver::from_conn_string("sqlite::memory:").await?;
    saver.setup().await?;

    let cfg = config_for("demo-thread");

    // First checkpoint — messages and counter at version 1
    let (cp1, vers1) = make_checkpoint(vec![
        ("messages", serde_json::json!(["hello"]), 1),
        ("counter", serde_json::json!(1), 1),
    ]);
    let metadata = CheckpointMetadata {
        source: Some(CheckpointSource::Loop),
        step: Some(0),
        ..Default::default()
    };
    let next_cfg = saver.aput(&cfg, &cp1, &metadata, &vers1).await?;
    println!("stored checkpoint #1: id={}", cp1.id);

    // Second checkpoint that references the first as parent — both
    // channels changed, so versions bump to 2.
    let (cp2, vers2) = make_checkpoint(vec![
        ("messages", serde_json::json!(["hello", "world"]), 2),
        ("counter", serde_json::json!(2), 2),
    ]);
    let metadata2 = CheckpointMetadata {
        source: Some(CheckpointSource::Loop),
        step: Some(1),
        ..Default::default()
    };
    saver.aput(&next_cfg, &cp2, &metadata2, &vers2).await?;
    println!("stored checkpoint #2: id={}", cp2.id);

    // Fetch latest
    let latest = saver.aget_tuple(&cfg).await?.expect("latest checkpoint");
    println!(
        "latest checkpoint id={} step={:?} channel_values={:?}",
        latest.checkpoint.id, latest.metadata.step, latest.checkpoint.channel_values
    );

    // List history (newest first)
    let history = saver.alist(Some(&cfg), None, None, None).await?;
    println!("history length = {}", history.len());
    for (i, t) in history.iter().enumerate() {
        println!("  [{}] id={} step={:?}", i, t.checkpoint.id, t.metadata.step);
    }

    // Store pending writes for the latest checkpoint
    let cfg_with_id: RunnableConfig = serde_json::from_value(serde_json::json!({
        "configurable": {
            "thread_id": "demo-thread",
            "checkpoint_ns": "",
            "checkpoint_id": cp2.id,
        }
    }))?;
    saver
        .aput_writes(
            &cfg_with_id,
            vec![(
                "outbox".into(),
                "task-1".into(),
                serde_json::json!({"sent": true}),
            )],
            "task-1".into(),
            "".into(),
        )
        .await?;
    let with_writes = saver.aget_tuple(&cfg_with_id).await?.unwrap();
    println!("pending writes = {:?}", with_writes.pending_writes);

    // Cleanup
    saver.adelete_thread("demo-thread".into()).await?;
    println!("thread deleted");

    Ok(())
}
