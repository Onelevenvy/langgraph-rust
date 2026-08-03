//! Benchmark: end-to-end Pregel loop hot paths.
//!
//! Run explicitly (it is `#[ignore]`d so normal `cargo test` runs stay fast):
//!
//! ```text
//! cargo test --release --test bench_pregel -- --ignored --nocapture
//! ```
//!
//! Note: this file installs mimalloc as the global allocator. On Windows the
//! default system heap is pathological for checkpointed workloads — large
//! (500KB+) per-super-step state blocks are allocated and freed each step, and
//! the heap's large-block path can slow the same binary by ~4x depending on
//! machine state, making results non-reproducible. mimalloc's size-classed,
//! thread-cached allocation keeps the numbers stable and representative of the
//! library's actual (allocation-bound) costs.
//!
//! Memory note: the growth benches use `LatestOnlySaver`, which retains only
//! the newest checkpoint per thread. `InMemorySaver` keeps every checkpoint, so
//! a growing-history run retains O(steps²) serialized state and OOMs after a
//! few hundred steps; latest-only keeps retention O(latest state) while
//! preserving the same per-step cost — each step still loads the newest
//! checkpoint and saves a fresh one. (The sqlite benches are disk-backed and
//! stay bounded at the caps below.)

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use langgraph::channels::{BinaryOperatorAggregate, Channel, LastValue};
use langgraph::checkpoint::config::RunnableConfigExt;
use langgraph::checkpoint::error::CheckpointError;
use langgraph::checkpoint::{
    BaseCheckpointSaver, ChannelVersions, Checkpoint, CheckpointMetadata, CheckpointTuple,
    InMemorySaver,
};
use langgraph::prelude::*;
use langgraph_checkpoint_sqlite::SqliteSaver;
use langgraph_prebuilt::add_messages_ref;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

fn make_message(i: usize) -> JsonValue {
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

/// (thread_id, checkpoint_ns) -> (checkpoint_id, checkpoint_json, metadata_json, parent_cid)
type StorageEntry = (String, JsonValue, JsonValue, Option<String>);
/// (thread_id, checkpoint_ns) -> the thread's newest checkpoint
type StorageMap = HashMap<(String, String), StorageEntry>;
/// (thread_id, checkpoint_ns, checkpoint_id) -> pending writes (interrupt-only path)
type WritesMap = HashMap<(String, String, String), Vec<(String, String, JsonValue)>>;

/// A checkpoint saver that retains only the newest checkpoint per thread.
///
/// `InMemorySaver` keeps every checkpoint forever, so a benchmark that runs `N`
/// super-steps with a growing message history retains O(N²) serialized state
/// and OOMs after a few hundred steps. Production savers prune; this one does
/// the same — each `put` replaces the thread's previous checkpoint, so retained
/// memory is O(latest state). The serde round-trip (to_value on `put`,
/// from_value on `get_tuple`) mirrors `InMemorySaver`, so the measured per-step
/// cost stays comparable.
struct LatestOnlySaver {
    storage: RwLock<StorageMap>,
    writes: RwLock<WritesMap>,
}

impl LatestOnlySaver {
    fn new() -> Self {
        Self {
            storage: RwLock::new(HashMap::new()),
            writes: RwLock::new(HashMap::new()),
        }
    }

    /// Number of retained checkpoints across all threads (1 per live thread).
    fn retained_count(&self) -> usize {
        self.storage.read().map(|s| s.len()).unwrap_or(0)
    }

    fn config_to_ids(config: &RunnableConfig) -> (String, String, Option<String>) {
        let configurable = config.get("configurable");
        let thread_id = configurable
            .and_then(|c| c.get("thread_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let checkpoint_ns = configurable
            .and_then(|c| c.get("checkpoint_ns"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let checkpoint_id = configurable
            .and_then(|c| c.get("checkpoint_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        (thread_id, checkpoint_ns, checkpoint_id)
    }
}

impl Default for LatestOnlySaver {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseCheckpointSaver for LatestOnlySaver {
    fn get_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        let (thread_id, checkpoint_ns, requested_id) = Self::config_to_ids(config);
        let storage = self.storage.read().unwrap();
        let Some((cid, checkpoint_json, metadata_json, parent_cid)) =
            storage.get(&(thread_id.clone(), checkpoint_ns.clone()))
        else {
            return Ok(None);
        };
        // A specific older checkpoint was requested — it's been pruned.
        if let Some(req) = &requested_id {
            if req != cid {
                return Ok(None);
            }
        }

        let checkpoint: Checkpoint = serde_json::from_value(checkpoint_json.clone())
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata: CheckpointMetadata = serde_json::from_value(metadata_json.clone())
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let parent_config = parent_cid.as_ref().map(|pid| {
            let mut c = RunnableConfig::new();
            c.insert(
                "configurable".to_string(),
                serde_json::json!({
                    "thread_id": thread_id.clone(),
                    "checkpoint_ns": checkpoint_ns.clone(),
                    "checkpoint_id": pid,
                }),
            );
            c
        });

        let pending_writes: Vec<(String, String, JsonValue)> = self
            .writes
            .read()
            .unwrap()
            .get(&(thread_id.clone(), checkpoint_ns.clone(), cid.clone()))
            .cloned()
            .unwrap_or_default();

        Ok(Some(CheckpointTuple {
            config: {
                let mut c = RunnableConfig::new();
                c.insert(
                    "configurable".to_string(),
                    serde_json::json!({
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cid,
                    }),
                );
                c
            },
            checkpoint,
            metadata,
            parent_config,
            pending_writes: if pending_writes.is_empty() {
                None
            } else {
                Some(pending_writes)
            },
        }))
    }

    fn list(
        &self,
        config: Option<&RunnableConfig>,
        _filter: Option<&HashMap<String, JsonValue>>,
        _before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        let (thread_id, checkpoint_ns) = match config {
            Some(c) => {
                let (tid, ns, _) = Self::config_to_ids(c);
                (tid, ns)
            }
            None => (String::new(), String::new()),
        };
        let storage = self.storage.read().unwrap();
        let mut entries: Vec<_> = storage
            .iter()
            .filter(|((tid, ns), _)| {
                (thread_id.is_empty() || tid == &thread_id)
                    && (checkpoint_ns.is_empty() || ns == &checkpoint_ns)
            })
            .collect();
        // Newest checkpoint id first, mirroring InMemorySaver's ordering.
        entries.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        if let Some(limit) = limit {
            entries.truncate(limit);
        }

        let mut results = Vec::new();
        for ((tid, ns), (cid, checkpoint_json, metadata_json, parent_cid)) in entries {
            let checkpoint: Checkpoint = serde_json::from_value(checkpoint_json.clone())
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            let metadata: CheckpointMetadata = serde_json::from_value(metadata_json.clone())
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            let parent_config = parent_cid.as_ref().map(|pid| {
                let mut c = RunnableConfig::new();
                c.insert(
                    "configurable".to_string(),
                    serde_json::json!({
                        "thread_id": tid,
                        "checkpoint_ns": ns,
                        "checkpoint_id": pid,
                    }),
                );
                c
            });
            results.push(CheckpointTuple {
                config: {
                    let mut c = RunnableConfig::new();
                    c.insert(
                        "configurable".to_string(),
                        serde_json::json!({
                            "thread_id": tid,
                            "checkpoint_ns": ns,
                            "checkpoint_id": cid,
                        }),
                    );
                    c
                },
                checkpoint,
                metadata,
                parent_config,
                pending_writes: None,
            });
        }
        Ok(results)
    }

    fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: &Checkpoint,
        metadata: &CheckpointMetadata,
        _new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        let (thread_id, checkpoint_ns, _) = Self::config_to_ids(config);

        let checkpoint_json = serde_json::to_value(checkpoint)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata_json =
            serde_json::to_value(metadata).map_err(|e| CheckpointError::Storage(e.to_string()))?;

        // The new checkpoint's parent is the thread's current newest.
        let parent_id = self
            .storage
            .read()
            .unwrap()
            .get(&(thread_id.clone(), checkpoint_ns.clone()))
            .map(|(cid, _, _, _)| cid.clone());

        // Replace the thread's checkpoint — only the newest is retained.
        self.storage.write().unwrap().insert(
            (thread_id.clone(), checkpoint_ns.clone()),
            (
                checkpoint.id.clone(),
                checkpoint_json,
                metadata_json,
                parent_id,
            ),
        );
        // Prune pending writes down to the newest checkpoint id.
        self.writes.write().unwrap().retain(|(tid, ns, cid), _| {
            tid != &thread_id || ns != &checkpoint_ns || cid == &checkpoint.id
        });

        let mut new_config = RunnableConfig::new();
        new_config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint.id,
            }),
        );
        Ok(new_config)
    }

    fn put_writes(
        &self,
        config: &RunnableConfig,
        writes: &[(String, String, JsonValue)],
        task_id: &str,
        _task_path: &str,
    ) -> Result<(), CheckpointError> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::config_to_ids(config);
        let checkpoint_id = checkpoint_id.unwrap_or_default();
        let mut writes_guard = self.writes.write().unwrap();
        let entry = writes_guard
            .entry((thread_id, checkpoint_ns, checkpoint_id))
            .or_default();
        for write in writes {
            entry.push((task_id.to_string(), write.1.clone(), write.2.clone()));
        }
        Ok(())
    }

    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        self.storage
            .write()
            .unwrap()
            .retain(|(tid, _), _| tid != thread_id);
        self.writes
            .write()
            .unwrap()
            .retain(|(tid, _, _), _| tid != thread_id);
        Ok(())
    }
}

/// Single-node graph: `messages` accumulates with the `add_messages` reducer.
/// Every invoke appends one message and saves a fresh checkpoint, so the
/// checkpointed state grows by one message per super-step.
fn build_linear_graph(checkpointer: Arc<dyn BaseCheckpointSaver>) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref)) as Box<dyn Channel>,
    );

    let mut graph = StateGraph::new(channels);
    graph
        .add_node(
            "append",
            |input: JsonValue, _config: RunnableConfig| async move {
                let n = input
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                Ok(json!({"messages": [make_message(n)]}))
            },
        )
        .unwrap();
    graph.add_edge(START, "append").unwrap();
    graph.add_edge("append", END).unwrap();

    graph
        .compile_builder()
        .checkpointer(checkpointer)
        .build()
        .unwrap()
}

/// Per-step cost of a checkpointed run as the message history grows.
///
/// If checkpointing re-serialized the full state every step this shows clear
/// super-linear growth (total work ~ O(steps^2)). Uses `LatestOnlySaver` so the
/// retained checkpoint set stays O(1) per thread instead of O(steps^2).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_linear_checkpoint_growth() {
    for steps in [100usize, 200, 400] {
        let saver = Arc::new(LatestOnlySaver::new());
        let app = build_linear_graph(saver.clone());
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-linear"}),
        );

        let start = Instant::now();
        for i in 0..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "linear checkpointed (latest-only): {steps:>4} steps, history->{steps} => {elapsed:?}  ({:?}/step) [retained checkpoints: {}]",
            elapsed / steps as u32,
            saver.retained_count()
        );
    }
}

/// Same growth benchmark against the SQLite saver — the persistent path where
/// incremental blob writes matter most.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_linear_checkpoint_sqlite() {
    for steps in [100usize, 200, 400] {
        let saver = SqliteSaver::from_conn_string("sqlite::memory:")
            .await
            .unwrap();
        saver.setup().await.unwrap();
        let app = build_linear_graph(Arc::new(saver));
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-sqlite"}),
        );

        let start = Instant::now();
        for i in 0..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "linear checkpointed (SQLite): {steps:>4} steps, history->{steps} => {elapsed:?}  ({:?}/step)",
            elapsed / steps as u32
        );
    }
}

/// Multi-super-step loop: one invoke runs exactly `target` super-steps via a
/// self-looping conditional edge, growing the history each step. This is the
/// shape that exercises the per-super-step output read (read_channels). The
/// `count` channel terminates the loop (see `build_loop_graph`); the recursion
/// limit is just a safety valve so a broken routing can't run forever.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_multi_step_loop() {
    for target in [50usize, 100, 200] {
        let saver = Arc::new(LatestOnlySaver::new());
        let app = build_loop_graph(saver.clone(), target);
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-loop"}),
        );
        // Safety valve: `target` can exceed the default recursion limit of 25.
        let config = config.with_recursion_limit(100_000);

        let input = json!({"count": 0, "messages": []});
        let t = Instant::now();
        app.ainvoke(&input, &config).await.unwrap();
        let elapsed = t.elapsed();
        // Sanity: the loop must actually run to `target` messages. get_state
        // reads the final checkpoint; a truncated run fails here, not silently.
        let snapshot = app.get_state(&config).unwrap();
        let got = snapshot
            .values
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len());
        assert_eq!(
            got,
            Some(target),
            "loop was silently truncated: expected {target} messages, got {got:?}"
        );
        println!(
            "multi-step loop: {target:>3} super-steps in one invoke => {elapsed:?}  ({:?}/super-step) [retained checkpoints: {}]",
            elapsed / target as u32,
            saver.retained_count()
        );
    }
}

/// Single-node graph with a self-loop: node "append" routes back to itself
/// until the history reaches `target` messages, then END.
///
/// The loop needs a plain `count` channel to terminate: conditional-edge
/// routing only sees the *node output* (the combined PregelNode evaluates
/// `branch.path` on the node's delta, not the accumulated state), so routing
/// on `messages.len()` always sees a single-element array and loops to the
/// recursion limit. The node emits the bumped counter alongside the new
/// message, and routing terminates on the counter.
fn build_loop_graph(
    checkpointer: Arc<dyn BaseCheckpointSaver>,
    target: usize,
) -> CompiledStateGraph {
    let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
    channels.insert(
        "messages".to_string(),
        Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref)) as Box<dyn Channel>,
    );
    channels.insert("count".to_string(), Box::new(LastValue::new("count")));

    let mut graph = StateGraph::new(channels);
    graph
        .add_node(
            "append",
            |input: JsonValue, _config: RunnableConfig| async move {
                let n = input.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
                Ok(json!({"count": n + 1, "messages": [make_message(n as usize)]}))
            },
        )
        .unwrap();
    graph.add_edge(START, "append").unwrap();
    graph
        .add_conditional_edges(
            "append",
            move |input: JsonValue, _config: RunnableConfig| async move {
                let n = input.get("count").and_then(|c| c.as_i64()).unwrap_or(0);
                Ok(json!(if n >= target as i64 { "END" } else { "again" }))
            },
            Some(HashMap::from([
                ("again".to_string(), "append".to_string()),
                ("END".to_string(), END.to_string()),
            ])),
        )
        .unwrap();

    graph
        .compile_builder()
        .checkpointer(checkpointer)
        .build()
        .unwrap()
}

/// Wall-clock time to run `branches` parallel nodes in a single super-step.
///
/// Every branch sleeps for a fixed time. A serial runner takes
/// `branches * sleep`; a parallel runner takes ~`sleep`.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn bench_parallel_fanout() {
    const BRANCH_SLEEP_MS: u64 = 40;

    for branches in [2usize, 4, 8] {
        let channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        let mut graph = StateGraph::new(channels);
        for i in 0..branches {
            graph
                .add_node(
                    format!("branch{i}"),
                    |_input: JsonValue, _config: RunnableConfig| async move {
                        tokio::time::sleep(Duration::from_millis(BRANCH_SLEEP_MS)).await;
                        Ok(json!({}))
                    },
                )
                .unwrap();
            graph.add_edge(START, format!("branch{i}")).unwrap();
        }
        let app = graph.compile().unwrap();

        let start = Instant::now();
        app.ainvoke(&json!({}), &RunnableConfig::new())
            .await
            .unwrap();
        let elapsed = start.elapsed();

        println!(
            "parallel fan-out: {branches} branches x {BRANCH_SLEEP_MS}ms => {elapsed:?}  (serial {BRANCH_SLEEP_MS}ms*n, parallel ~{BRANCH_SLEEP_MS}ms)"
        );
    }
}

/// Linear chain of `nodes` no-op nodes (START -> n0 -> n1 -> ... -> END),
/// mirroring juncture's `sequential.rs` bench 1:1 so the per-node framework
/// overhead is directly comparable (juncture reference: ~2.3µs/node at 1000,
/// measured 2026-08-02). Each node returns an empty update; the last writes a
/// `done` marker so the chain is verified to have run all `nodes` super-steps —
/// a silent truncation fails the assert, not the timing. A chain is one
/// super-step per node, so the recursion limit is raised above the 25 default.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_sequential_chain() {
    const REPS: u32 = 3;

    for nodes in [10usize, 100, 500, 1000] {
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert(
            "messages".to_string(),
            Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref))
                as Box<dyn Channel>,
        );
        channels.insert("done".to_string(), Box::new(LastValue::new("done")));

        let mut graph = StateGraph::new(channels);
        let names: Vec<String> = (0..nodes).map(|i| format!("node_{i}")).collect();
        for (i, name) in names.iter().enumerate() {
            let is_last = i + 1 == nodes;
            graph
                .add_node(
                    name.clone(),
                    move |_input: JsonValue, _config: RunnableConfig| {
                        let is_last = is_last;
                        async move {
                            Ok(if is_last {
                                json!({"done": true})
                            } else {
                                json!({})
                            })
                        }
                    },
                )
                .unwrap();
        }
        graph.add_edge(START, names[0].clone()).unwrap();
        for i in 0..nodes - 1 {
            graph
                .add_edge(names[i].clone(), names[i + 1].clone())
                .unwrap();
        }
        graph.add_edge(names[nodes - 1].clone(), END).unwrap();
        let app = graph.compile().unwrap();

        let config = RunnableConfig::new().with_recursion_limit(100_000);

        let mut best = Duration::MAX;
        for _ in 0..REPS {
            let t = Instant::now();
            let output = app.ainvoke(&json!({}), &config).await.unwrap();
            best = best.min(t.elapsed());
            assert_eq!(
                output.get("done"),
                Some(&json!(true)),
                "chain was truncated: expected all {nodes} nodes to run"
            );
        }
        println!(
            "sequential chain: {nodes:>4} no-op nodes => {best:?}  ({:?}/node)",
            best / nodes as u32
        );
    }
}

/// The win case for incremental writes: a large static channel (e.g. embedded
/// knowledge base) written once at thread start, then untouched while
/// `messages` grows every step. Without delta writes the static channel's
/// blob gets re-encoded and re-inserted every single step.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn bench_sqlite_static_context() {
    const CONTEXT_SIZE: usize = 128 * 1024;

    for steps in [100usize, 200, 400] {
        let saver = SqliteSaver::from_conn_string("sqlite::memory:")
            .await
            .unwrap();
        saver.setup().await.unwrap();

        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert(
            "messages".to_string(),
            Box::new(BinaryOperatorAggregate::new("messages", add_messages_ref))
                as Box<dyn Channel>,
        );
        channels.insert(
            "context".to_string(),
            Box::new(LastValue::new("context")) as Box<dyn Channel>,
        );

        let mut graph = StateGraph::new(channels);
        graph
            .add_node(
                "append",
                |input: JsonValue, _config: RunnableConfig| async move {
                    let n = input
                        .get("messages")
                        .and_then(|m| m.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    Ok(json!({"messages": [make_message(n)]}))
                },
            )
            .unwrap();
        graph.add_edge(START, "append").unwrap();
        graph.add_edge("append", END).unwrap();
        let app = graph
            .compile_builder()
            .checkpointer(Arc::new(saver))
            .build()
            .unwrap();

        let context = "x".repeat(CONTEXT_SIZE);
        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            json!({"thread_id": "bench-ctx"}),
        );

        // Seed the static channel once, then grow messages every step.
        app.ainvoke(
            &json!({"messages": [make_message(0)], "context": context}),
            &config,
        )
        .await
        .unwrap();

        let start = Instant::now();
        for i in 1..steps {
            let input = json!({"messages": [make_message(i)]});
            app.ainvoke(&input, &config).await.unwrap();
        }
        let elapsed = start.elapsed();
        println!(
            "sqlite static context: {steps:>4} steps, {CONTEXT_SIZE}-byte static channel => {elapsed:?}  ({:?}/step)",
            elapsed / (steps - 1) as u32
        );
    }
}

/// Guards against the benchmark graphs silently becoming no-ops.
#[tokio::test]
async fn sanity_bench_graphs_work() {
    let app = build_linear_graph(Arc::new(InMemorySaver::new()));
    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        json!({"thread_id": "bench-sanity"}),
    );
    app.ainvoke(&json!({"messages": [make_message(0)]}), &config)
        .await
        .unwrap();
    app.ainvoke(&json!({"messages": [make_message(1)]}), &config)
        .await
        .unwrap();
    let snapshot = app.get_state(&config).unwrap();
    assert_eq!(
        snapshot
            .values
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|a| a.len()),
        Some(4)
    );
}
