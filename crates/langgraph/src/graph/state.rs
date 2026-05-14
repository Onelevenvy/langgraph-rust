use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use langgraph_checkpoint::config::{RunnableConfig, RunnableConfigExt};
use langgraph_checkpoint::cache::base::BaseCache;
use langgraph_checkpoint::store::base::BaseStore;
use langgraph_checkpoint::checkpoint::base::BaseCheckpointSaver;
use crate::channels::{Channel, EphemeralValue, NamedBarrierValue};
use crate::constants::{START, END, RESUME, INTERRUPT, NULL_TASK_ID};
use crate::runnable::{Runnable, RunnableError, IntoNodeFunction};
use crate::graph::node::StateNodeSpec;
use crate::graph::branch::BranchSpec;
use crate::pregel::{PregelNode, PregelRunner, ChannelVersions, channels_from_checkpoint, PregelExecutableTask};
use crate::pregel::algo::{prepare_next_tasks, apply_writes};
use crate::pregel::io::{map_input, map_command, read_channels};
use crate::stream::StreamPart;
use crate::types::{Command, StreamMode, StateSnapshot, PregelTask, Interrupt};
use langgraph_checkpoint::checkpoint::types::CheckpointMetadata;

/// Multi-source edge: waits for all sources to complete before routing to target.
type WaitingEdge = (Vec<String>, String);

/// Error type for graph building operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node '{0}' already exists")]
    DuplicateNode(String),

    #[error("unknown node '{0}'")]
    UnknownNode(String),

    #[error("cannot use reserved name '{0}'")]
    ReservedName(String),

    #[error("START cannot be an edge target")]
    StartAsTarget,

    #[error("END cannot be an edge source")]
    EndAsSource,

    #[error("no outgoing edge from START")]
    NoStartEdge,

    #[error("graph validation failed: {0}")]
    ValidationError(String),

    #[error(transparent)]
    Runnable(#[from] RunnableError),

    #[error("checkpoint error: {0}")]
    Checkpoint(String),
}

/// Builder for constructing a state graph.
///
/// `S` is the state type (typically a struct with `#[derive(StateGraph)]`).
/// Channels are derived from `S::create_channels()` (the derive macro generates this).
///
/// # Example
/// ```rust,ignore
/// use langgraph::prelude::*;
///
/// let mut graph = StateGraph::new(channels);
/// graph.add_node("agent", agent_fn);
/// graph.add_edge(START, "agent");
/// graph.add_edge("agent", END);
/// let compiled = graph.compile(checkpointer, None, None, None, None, false, None);
/// ```
pub struct StateGraph {
    /// Registered nodes keyed by name.
    nodes: HashMap<String, StateNodeSpec>,
    /// Simple directed edges: (source, target).
    edges: HashSet<(String, String)>,
    /// Multi-source "join" edges: ([source1, source2, ...], target).
    waiting_edges: HashSet<WaitingEdge>,
    /// Conditional edges: source -> branch_name -> BranchSpec.
    branches: HashMap<String, HashMap<String, BranchSpec>>,
    /// Channels derived from the state schema.
    channels: HashMap<String, Box<dyn Channel>>,
    /// Whether compile() has been called.
    compiled: bool,
}

impl StateGraph {
    /// Create a new StateGraph with the given channels.
    ///
    /// Typically called via the derive macro: `MyState::create_channels()`.
    pub fn new(channels: HashMap<String, Box<dyn Channel>>) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashSet::new(),
            waiting_edges: HashSet::new(),
            branches: HashMap::new(),
            channels,
            compiled: false,
        }
    }

    /// Add a node to the graph.
    ///
    /// Accepts async closures (the default), sync closures via `node_fn!()` or `SyncNodeFn`,
    /// or pre-built `Arc<dyn Runnable>`.
    ///
    /// # Examples
    /// ```ignore
    /// // Async closure (default)
    /// graph.add_node("agent", |input, _config| async move {
    ///     Ok(json!({"result": "done"}))
    /// })?;
    ///
    /// // Sync closure via node_fn! macro
    /// graph.add_node("doubler", node_fn!(|input, _config| {
    ///     let n = input.as_i64().unwrap_or(0);
    ///     Ok(json!(n * 2))
    /// }))?;
    /// ```
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        action: impl IntoNodeFunction,
    ) -> Result<&mut Self, GraphError> {
        let name = name.into();
        self.validate_node_name(&name)?;
        let runnable = action.into_runnable(&name);
        self.nodes.insert(name.clone(), StateNodeSpec::new(name, runnable));
        Ok(self)
    }

    /// Add a direct edge from `start` to `end`.
    ///
    /// `start` can be a node name or `START`.
    /// `end` can be a node name or `END`.
    pub fn add_edge(
        &mut self,
        start: impl Into<String>,
        end: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let start = start.into();
        let end = end.into();

        if start == END {
            return Err(GraphError::EndAsSource);
        }
        if end == START {
            return Err(GraphError::StartAsTarget);
        }
        if start != START && !self.nodes.contains_key(&start) {
            return Err(GraphError::UnknownNode(start));
        }
        if end != END && !self.nodes.contains_key(&end) {
            return Err(GraphError::UnknownNode(end));
        }

        self.edges.insert((start, end));
        Ok(self)
    }

    /// Add a multi-source join edge.
    ///
    /// The graph waits for ALL `starts` to complete before routing to `end`.
    pub fn add_join_edge(
        &mut self,
        starts: Vec<String>,
        end: impl Into<String>,
    ) -> Result<&mut Self, GraphError> {
        let end = end.into();
        if end == START {
            return Err(GraphError::StartAsTarget);
        }
        for s in &starts {
            if s == END {
                return Err(GraphError::EndAsSource);
            }
            if s != START && !self.nodes.contains_key(s) {
                return Err(GraphError::UnknownNode(s.clone()));
            }
        }
        if end != END && !self.nodes.contains_key(&end) {
            return Err(GraphError::UnknownNode(end));
        }
        self.waiting_edges.insert((starts, end));
        Ok(self)
    }

    /// Add conditional edges from `source`.
    ///
    /// The `path` function evaluates the state and returns a routing key.
    /// The `path_map` maps routing keys to destination node names.
    /// If `path_map` is `None`, the routing key is used directly as the node name.
    pub fn add_conditional_edges(
        &mut self,
        source: impl Into<String>,
        path: impl IntoNodeFunction,
        path_map: Option<HashMap<String, String>>,
    ) -> Result<&mut Self, GraphError> {
        let source = source.into();
        if source != START && !self.nodes.contains_key(&source) {
            return Err(GraphError::UnknownNode(source));
        }

        let branch_name = format!("branch:{}", source);
        let runnable = path.into_runnable(&branch_name);
        let branch = BranchSpec::new(runnable, path_map);

        self.branches
            .entry(source)
            .or_default()
            .insert(branch_name, branch);

        Ok(self)
    }

    /// Set the entry point (equivalent to `add_edge(START, key)`).
    pub fn set_entry_point(&mut self, key: impl Into<String>) -> Result<&mut Self, GraphError> {
        self.add_edge(START, key)
    }

    /// Set the finish point (equivalent to `add_edge(key, END)`).
    pub fn set_finish_point(&mut self, key: impl Into<String>) -> Result<&mut Self, GraphError> {
        self.add_edge(key, END)
    }

    /// Compile the graph into an executable `CompiledStateGraph`.
    ///
    /// The compiled graph implements `Runnable` and can be invoked with state.
    /// Uses all defaults (no checkpointer, no cache, no store, etc.).
    ///
    /// For custom configuration, use `compile_builder()`.
    pub fn compile(&mut self) -> Result<CompiledStateGraph, GraphError> {
        self.compile_with(None, None, None, None, None, false, None, None)
    }

    /// Start building compile options with a builder pattern.
    ///
    /// # Example
    /// ```ignore
    /// let compiled = graph.compile_builder()
    ///     .debug(true)
    ///     .name("my_graph")
    ///     .build()?;
    /// ```
    pub fn compile_builder(&mut self) -> CompileBuilder<'_> {
        CompileBuilder {
            graph: self,
            checkpointer: None,
            cache: None,
            store: None,
            interrupt_before: None,
            interrupt_after: None,
            debug: false,
            name: None,
            recursion_limit: None,
        }
    }

    /// Internal: compile with explicit parameters.
    fn compile_with(
        &mut self,
        checkpointer: Option<Arc<dyn BaseCheckpointSaver>>,
        cache: Option<Arc<dyn BaseCache>>,
        store: Option<Arc<dyn BaseStore>>,
        interrupt_before: Option<Vec<String>>,
        interrupt_after: Option<Vec<String>>,
        debug: bool,
        name: Option<String>,
        recursion_limit: Option<u64>,
    ) -> Result<CompiledStateGraph, GraphError> {
        self.validate()?;

        // Add START channel (ephemeral)
        self.channels.insert(
            START.to_string(),
            Box::new(EphemeralValue::new(START, false)),
        );

        // Add trigger channels for each node ("branch:to:{name}")
        for name in self.nodes.keys() {
            let trigger_key = format!("branch:to:{}", name);
            self.channels
                .insert(trigger_key.clone(), Box::new(EphemeralValue::new(trigger_key, false)));
        }

        // Add barrier channels for waiting edges
        for (sources, target) in &self.waiting_edges {
            let barrier_name = format!("join:{}:{}", sources.join("+"), target);
            let names: HashSet<String> = sources.iter().cloned().collect();
            self.channels.insert(
                barrier_name.clone(),
                Box::new(NamedBarrierValue::new(barrier_name, names)),
            );
        }

        self.compiled = true;

        let channels = self.channels
            .iter()
            .map(|(k, c)| (k.clone(), c.clone_channel()))
            .collect();

        Ok(CompiledStateGraph {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            waiting_edges: self.waiting_edges.clone(),
            branches: self.branches.clone(),
            channels,
            checkpointer,
            cache,
            store,
            interrupt_before: interrupt_before.unwrap_or_default(),
            interrupt_after: interrupt_after.unwrap_or_default(),
            debug,
            name: name.unwrap_or_else(|| "StateGraph".to_string()),
            recursion_limit: recursion_limit.unwrap_or(DEFAULT_RECURSION_LIMIT),
        })
    }

    fn validate_node_name(&self, name: &str) -> Result<(), GraphError> {
        if name == START || name == END {
            return Err(GraphError::ReservedName(name.to_string()));
        }
        if self.nodes.contains_key(name) {
            return Err(GraphError::DuplicateNode(name.to_string()));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), GraphError> {
        // START must have at least one outgoing edge
        let has_start_edge = self.edges.iter().any(|(s, _)| s == START)
            || self.waiting_edges.iter().any(|(s, _)| s.contains(&START.to_string()))
            || self.branches.contains_key(START);
        if !has_start_edge {
            return Err(GraphError::NoStartEdge);
        }

        // Validate all edge endpoints exist
        for (start, end) in &self.edges {
            if start != START && !self.nodes.contains_key(start) {
                return Err(GraphError::UnknownNode(start.clone()));
            }
            if end != END && !self.nodes.contains_key(end) {
                return Err(GraphError::UnknownNode(end.clone()));
            }
        }

        Ok(())
    }
}

/// Builder for configuring `compile()` options.
pub struct CompileBuilder<'a> {
    graph: &'a mut StateGraph,
    checkpointer: Option<Arc<dyn BaseCheckpointSaver>>,
    cache: Option<Arc<dyn BaseCache>>,
    store: Option<Arc<dyn BaseStore>>,
    interrupt_before: Option<Vec<String>>,
    interrupt_after: Option<Vec<String>>,
    debug: bool,
    name: Option<String>,
    recursion_limit: Option<u64>,
}

impl<'a> CompileBuilder<'a> {
    pub fn checkpointer(mut self, cp: Arc<dyn BaseCheckpointSaver>) -> Self {
        self.checkpointer = Some(cp);
        self
    }

    pub fn cache(mut self, cache: Arc<dyn BaseCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn store(mut self, store: Arc<dyn BaseStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn interrupt_before(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_before = Some(nodes);
        self
    }

    pub fn interrupt_after(mut self, nodes: Vec<String>) -> Self {
        self.interrupt_after = Some(nodes);
        self
    }

    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn recursion_limit(mut self, limit: u64) -> Self {
        self.recursion_limit = Some(limit);
        self
    }

    pub fn build(self) -> Result<CompiledStateGraph, GraphError> {
        self.graph.compile_with(
            self.checkpointer,
            self.cache,
            self.store,
            self.interrupt_before,
            self.interrupt_after,
            self.debug,
            self.name,
            self.recursion_limit,
        )
    }
}

/// A compiled, executable state graph.
///
/// This is the result of `StateGraph::compile()` and implements `Runnable`.
/// In the full Pregel engine (Phase 6), this will execute in BSP super-steps.
/// Currently it provides a simplified sequential execution model.
pub struct CompiledStateGraph {
    nodes: HashMap<String, StateNodeSpec>,
    edges: HashSet<(String, String)>,
    waiting_edges: HashSet<WaitingEdge>,
    branches: HashMap<String, HashMap<String, BranchSpec>>,
    channels: HashMap<String, Box<dyn Channel>>,
    checkpointer: Option<Arc<dyn BaseCheckpointSaver>>,
    #[allow(dead_code)]
    cache: Option<Arc<dyn BaseCache>>,
    store: Option<Arc<dyn BaseStore>>,
    interrupt_before: Vec<String>,
    interrupt_after: Vec<String>,
    debug: bool,
    name: String,
    recursion_limit: u64,
}

impl CompiledStateGraph {
    /// Get the node names in this graph.
    pub fn node_names(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get the channel names in this graph.
    pub fn channel_names(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }

    /// Check if a node exists.
    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Get the graph name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the checkpointer, if any.
    pub fn checkpointer(&self) -> Option<&Arc<dyn BaseCheckpointSaver>> {
        self.checkpointer.as_ref()
    }

    /// Get the store, if any.
    pub fn store(&self) -> Option<&Arc<dyn BaseStore>> {
        self.store.as_ref()
    }

    /// Save a checkpoint from current channel state.
    fn save_checkpoint(
        &self,
        checkpointer: &Arc<dyn BaseCheckpointSaver>,
        config: &RunnableConfig,
        channels: &HashMap<String, Box<dyn Channel>>,
        channel_versions: &ChannelVersions,
        versions_seen: &HashMap<String, HashMap<String, JsonValue>>,
    ) -> Option<RunnableConfig> {
        use langgraph_checkpoint::checkpoint::id::uuid6;
        use chrono::Utc;

        // Collect all channel values (including trigger channels for state history)
        let channel_values: HashMap<String, JsonValue> = channels
            .iter()
            .filter_map(|(k, v)| v.checkpoint().map(|val| (k.clone(), val)))
            .collect();

        let checkpoint = langgraph_checkpoint::Checkpoint {
            v: 2,
            id: uuid6(),
            ts: Utc::now().to_rfc3339(),
            channel_values,
            channel_versions: channel_versions.clone(),
            versions_seen: versions_seen.clone(),
            updated_channels: None,
        };

        let metadata = CheckpointMetadata::default();
        checkpointer.put(config, &checkpoint, &metadata, channel_versions).ok()
    }

    /// Determine which nodes should execute next given the current state.
    ///
    /// This is the "plan" phase of the BSP cycle.
    pub fn get_next_nodes(&self, state: &HashMap<String, JsonValue>) -> Vec<String> {
        let mut next = Vec::new();

        // Check which nodes are triggered by edges from completed nodes
        for (start, end) in &self.edges {
            if (start == START || state.contains_key(&format!("branch:to:{}", start)))
                && end != END {
                    next.push(end.clone());
                }
        }

        // Check conditional branches
        for (source, branches) in &self.branches {
            if source == START || state.contains_key(&format!("branch:to:{}", source)) {
                for _branch in branches.values() {
                    // Evaluate the branch path to determine routing
                    // For now, we'd need to actually invoke the path runnable
                    // This is handled by the Pregel engine in Phase 6
                }
            }
        }

        next
    }

    /// Get the current state of the graph from the checkpointer.
    ///
    /// Returns a `StateSnapshot` containing the current channel values,
    /// the names of nodes that will execute next, pending tasks, and
    /// any unresolved interrupts.
    ///
    /// Requires a checkpointer to be configured.
    ///
    /// # Example
    /// ```ignore
    /// let snapshot = compiled.get_state(&config)?;
    /// println!("next: {:?}", snapshot.next);
    /// println!("values: {}", snapshot.values);
    /// ```
    pub fn get_state(&self, config: &RunnableConfig) -> Result<StateSnapshot, GraphError> {
        let checkpointer = self.checkpointer.as_ref().ok_or_else(|| {
            GraphError::ValidationError("No checkpointer set".to_string())
        })?;

        let saved = checkpointer
            .get_tuple(config)
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;

        let Some(saved) = saved else {
            return Ok(StateSnapshot {
                values: JsonValue::Object(serde_json::Map::new()),
                next: vec![],
                config: config.clone(),
                metadata: None,
                created_at: None,
                parent_config: None,
                tasks: vec![],
                interrupts: vec![],
            });
        };

        // Reconstruct channels from checkpoint
        let cp_channels: HashMap<String, Option<JsonValue>> = saved
            .checkpoint
            .channel_values
            .iter()
            .map(|(k, v)| (k.clone(), Some(v.clone())))
            .collect();
        let mut channels = channels_from_checkpoint(&self.channels, &cp_channels);

        let mut channel_versions = saved.checkpoint.channel_versions.clone();
        let mut versions_seen = saved.checkpoint.versions_seen.clone();

        // Apply null-task pending writes (input writes not tied to a task)
        if let Some(ref pending) = saved.pending_writes {
            for (tid, chan, val) in pending {
                if tid == NULL_TASK_ID {
                    if let Some(ch) = channels.get(chan) {
                        ch.update(&[val.clone()]).ok();
                    }
                }
            }
        }

        // Build PregelNode specs and prepare next tasks
        let pregel_nodes = build_pregel_nodes(
            &self.nodes,
            &self.edges,
            &self.waiting_edges,
            &self.branches,
            &self.channels,
        );
        let trigger_to_nodes = crate::pregel::build_trigger_to_nodes(&pregel_nodes);

        let step = 0u64;
        let checkpoint_id = format!("{:032}", step);
        let pending_writes: Vec<(String, String, JsonValue)> = saved
            .pending_writes
            .as_ref()
            .map(|pw| pw.to_vec())
            .unwrap_or_default();

        let mut tasks = prepare_next_tasks(
            &pregel_nodes,
            &channels,
            config,
            step,
            &mut versions_seen,
            &trigger_to_nodes,
            None,
            &checkpoint_id,
            &pending_writes,
            &channel_versions,
        );

        // Apply non-INTERRUPT, non-ERROR pending writes to tasks
        // so that the snapshot values reflect completed task outputs
        if let Some(ref pending) = saved.pending_writes {
            for (tid, chan, val) in pending {
                if chan == INTERRUPT || chan == crate::constants::ERROR {
                    continue;
                }
                if tid == NULL_TASK_ID {
                    continue;
                }
                if let Some(task) = tasks.iter_mut().find(|t| &t.id == tid) {
                    task.writes.push((chan.clone(), val.clone()));
                }
            }
        }

        // Apply writes from completed tasks to get final channel state
        apply_writes(
            &mut channels,
            &tasks,
            &mut versions_seen,
            &mut channel_versions,
            &trigger_to_nodes,
            |current| {
                let num = current
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                JsonValue::String(format!("{:032}", num + 1))
            },
        );

        // Read channel values
        let output_keys: Vec<String> = channels
            .keys()
            .filter(|k| !k.starts_with("branch:") && !k.starts_with("join:") && *k != START)
            .cloned()
            .collect();
        let values = read_channels(&channels, &output_keys);

        // Build next: names of tasks that have NOT written yet
        let next: Vec<String> = tasks
            .iter()
            .filter(|t| t.writes.is_empty())
            .map(|t| t.name.clone())
            .collect();

        // Extract interrupts from pending writes
        let interrupts: Vec<Interrupt> = saved
            .pending_writes
            .as_ref()
            .map(|pw| {
                pw.iter()
                    .filter(|(_, chan, _)| chan == INTERRUPT)
                    .filter_map(|(_, _, val)| {
                        serde_json::from_value::<Interrupt>(val.clone()).ok()
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Build PregelTask list for the snapshot
        let snapshot_tasks: Vec<PregelTask> = tasks
            .iter()
            .map(|t| {
                let task_interrupts: Vec<Interrupt> = saved
                    .pending_writes
                    .as_ref()
                    .map(|pw| {
                        pw.iter()
                            .filter(|(tid, chan, _)| tid == &t.id && chan == INTERRUPT)
                            .filter_map(|(_, _, val)| {
                                serde_json::from_value::<Interrupt>(val.clone()).ok()
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                PregelTask {
                    id: t.id.clone(),
                    name: t.name.clone(),
                    path: vec![],
                    error: None,
                    interrupts: task_interrupts,
                    result: None,
                }
            })
            .collect();

        Ok(StateSnapshot {
            values,
            next,
            config: saved.config.clone(),
            metadata: Some(saved.metadata.clone()),
            created_at: Some(saved.checkpoint.ts.clone()),
            parent_config: saved.parent_config.clone(),
            tasks: snapshot_tasks,
            interrupts,
        })
    }

    /// Manually update the graph state.
    ///
    /// Applies the given values to the current checkpoint's channels and
    /// saves a new checkpoint. This allows updating custom state fields
    /// (like `name`, `birthday`) outside of normal node execution.
    ///
    /// Requires a checkpointer to be configured.
    ///
    /// # Arguments
    /// * `config` - The runnable config (must include `thread_id`)
    /// * `values` - A JSON object of channel updates, e.g. `{"name": "LangGraph"}`
    ///
    /// # Example
    /// ```ignore
    /// compiled.update_state(&config, json!({"name": "LangGraph (library)"}))?;
    /// let snapshot = compiled.get_state(&config)?;
    /// assert_eq!(snapshot.values["name"], "LangGraph (library)");
    /// ```
    pub fn update_state(
        &self,
        config: &RunnableConfig,
        values: &JsonValue,
    ) -> Result<RunnableConfig, GraphError> {
        let checkpointer = self.checkpointer.as_ref().ok_or_else(|| {
            GraphError::ValidationError("No checkpointer set".to_string())
        })?;

        let saved = checkpointer
            .get_tuple(config)
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;

        // Reconstruct channels from checkpoint (or fresh if none)
        let channels: HashMap<String, Box<dyn Channel>> = if let Some(ref saved) = saved {
            let cp_channels: HashMap<String, Option<JsonValue>> = saved
                .checkpoint
                .channel_values
                .iter()
                .map(|(k, v)| (k.clone(), Some(v.clone())))
                .collect();
            channels_from_checkpoint(&self.channels, &cp_channels)
        } else {
            self.channels
                .iter()
                .map(|(k, c)| (k.clone(), c.clone_channel()))
                .collect()
        };

        let mut channel_versions = saved
            .as_ref()
            .map(|s| s.checkpoint.channel_versions.clone())
            .unwrap_or_default();
        let versions_seen = saved
            .as_ref()
            .map(|s| s.checkpoint.versions_seen.clone())
            .unwrap_or_default();

        // Apply the update values to channels
        if let Some(obj) = values.as_object() {
            for (key, val) in obj {
                if let Some(ch) = channels.get(key) {
                    ch.update(&[val.clone()]).ok();
                    // Bump the channel version
                    let new_version = channel_versions
                        .get(key)
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0)
                        + 1;
                    channel_versions.insert(
                        key.clone(),
                        JsonValue::String(format!("{:032}", new_version)),
                    );
                }
            }
        }

        // Save the updated checkpoint
        self.save_checkpoint(checkpointer, config, &channels, &channel_versions, &versions_seen);

        Ok(config.clone())
    }

    /// Get the state history (all checkpoints) for a thread.
    ///
    /// Returns a list of `StateSnapshot` in reverse chronological order
    /// (newest first). Each snapshot contains the checkpoint's channel values,
    /// which node would execute next, and metadata.
    ///
    /// This enables "time travel" — reviewing past states and resuming
    /// from any checkpoint.
    ///
    /// # Example
    /// ```ignore
    /// let history = compiled.get_state_history(&config)?;
    /// for snapshot in &history {
    ///     println!("messages: {}, next: {:?}", snapshot.values["messages"].as_array().map(|a| a.len()), snapshot.next);
    /// }
    /// ```
    pub fn get_state_history(&self, config: &RunnableConfig) -> Result<Vec<StateSnapshot>, GraphError> {
        let checkpointer = self.checkpointer.as_ref().ok_or_else(|| {
            GraphError::ValidationError("No checkpointer set".to_string())
        })?;

        let tuples = checkpointer
            .list(Some(config), None, None, None)
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;

        let mut snapshots = Vec::new();

        // Build PregelNode specs for task preparation
        let pregel_nodes = build_pregel_nodes(
            &self.nodes,
            &self.edges,
            &self.waiting_edges,
            &self.branches,
            &self.channels,
        );
        let trigger_to_nodes = crate::pregel::build_trigger_to_nodes(&pregel_nodes);

        for saved in &tuples {
            // Reconstruct channels from checkpoint
            let cp_channels: HashMap<String, Option<JsonValue>> = saved
                .checkpoint
                .channel_values
                .iter()
                .map(|(k, v)| (k.clone(), Some(v.clone())))
                .collect();
            let channels = channels_from_checkpoint(&self.channels, &cp_channels);

            let channel_versions = saved.checkpoint.channel_versions.clone();
            let mut versions_seen = saved.checkpoint.versions_seen.clone();

            // Apply non-INTERRUPT pending writes to get the correct channel state
            if let Some(ref pending) = saved.pending_writes {
                for (tid, chan, val) in pending {
                    if chan == INTERRUPT || chan == crate::constants::ERROR {
                        continue;
                    }
                    if tid == NULL_TASK_ID {
                        if let Some(ch) = channels.get(chan) {
                            ch.update(&[val.clone()]).ok();
                        }
                        continue;
                    }
                    if let Some(ch) = channels.get(chan) {
                        ch.update(&[val.clone()]).ok();
                    }
                }
            }

            // Read output values
            let output_keys: Vec<String> = channels
                .keys()
                .filter(|k| !k.starts_with("branch:") && !k.starts_with("join:") && *k != START)
                .cloned()
                .collect();
            let values = read_channels(&channels, &output_keys);

            // Prepare tasks to determine what would execute next
            let checkpoint_id = saved.checkpoint.id.clone();
            let pending_writes: Vec<(String, String, JsonValue)> = saved
                .pending_writes
                .as_ref()
                .map(|pw| pw.iter().map(|(t, c, v)| (t.clone(), c.clone(), v.clone())).collect())
                .unwrap_or_default();

            let tasks = prepare_next_tasks(
                &pregel_nodes,
                &channels,
                &RunnableConfig::new(),
                0,
                &mut versions_seen,
                &trigger_to_nodes,
                None,
                &checkpoint_id,
                &pending_writes,
                &channel_versions,
            );

            // Next = tasks that haven't written yet
            let next: Vec<String> = tasks
                .iter()
                .filter(|t| t.writes.is_empty())
                .map(|t| t.name.clone())
                .collect();

            // Extract interrupts from pending writes
            let interrupts: Vec<Interrupt> = saved
                .pending_writes
                .as_ref()
                .map(|pw| {
                    pw.iter()
                        .filter(|(_, chan, _)| chan == INTERRUPT)
                        .filter_map(|(_, _, val)| {
                            serde_json::from_value::<Interrupt>(val.clone()).ok()
                        })
                        .collect()
                })
                .unwrap_or_default();

            snapshots.push(StateSnapshot {
                values,
                next,
                config: saved.config.clone(),
                metadata: Some(saved.metadata.clone()),
                created_at: Some(saved.checkpoint.ts.clone()),
                parent_config: saved.parent_config.clone(),
                tasks: vec![],
                interrupts,
            });
        }

        Ok(snapshots)
    }
}

impl Clone for CompiledStateGraph {
    fn clone(&self) -> Self {
        let channels: HashMap<String, Box<dyn Channel>> = self.channels
            .iter()
            .map(|(k, c)| (k.clone(), c.clone_channel()))
            .collect();

        // Manually clone branches (nested HashMap with non-Clone inner values already handled by Arc)
        let branches: HashMap<String, HashMap<String, BranchSpec>> = self.branches
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            waiting_edges: self.waiting_edges.clone(),
            branches,
            channels,
            checkpointer: self.checkpointer.clone(),
            cache: self.cache.clone(),
            store: self.store.clone(),
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
            debug: self.debug,
            name: self.name.clone(),
            recursion_limit: self.recursion_limit,
        }
    }
}

/// Build PregelNode specs from the graph structure.
///
/// For each node, creates a combined runnable that:
/// 1. Executes the node logic
/// 2. Writes state updates to channels
/// 3. Writes to trigger / barrier channels for edge targets
///
/// Join edges (from `add_join_edge`) use a `NamedBarrierValue` channel
/// (named `join:{sources}:{target}`) instead of a plain `branch:to:{target}`.
/// Each source node writes its own name into the barrier channel; the barrier
/// becomes available only when ALL sources have written, at which point the
/// join-target node is triggered.
fn build_pregel_nodes(
    nodes: &HashMap<String, StateNodeSpec>,
    edges: &HashSet<(String, String)>,
    waiting_edges: &HashSet<WaitingEdge>,
    branches: &HashMap<String, HashMap<String, BranchSpec>>,
    channels: &HashMap<String, Box<dyn Channel>>,
) -> HashMap<String, PregelNode> {
    let mut pregel_nodes = HashMap::new();

    // Build a map of source -> [plain-edge targets] (excluding END)
    let mut edge_targets: HashMap<String, Vec<String>> = HashMap::new();
    for (start, end) in edges {
        if end != END {
            edge_targets.entry(start.clone()).or_default().push(end.clone());
        }
    }

    // Build join-edge lookup maps from waiting_edges:
    //
    //   join_writes_for_source:  source_name -> [(barrier_channel_name, source_name)]
    //     When a source node completes, it writes its own name into every
    //     barrier channel it participates in.
    //
    //   join_trigger_for_target: target_name -> barrier_channel_name
    //     The join-target node uses the barrier channel as its sole trigger
    //     instead of the default "branch:to:{name}" ephemeral channel.
    let mut join_writes_for_source: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut join_trigger_for_target: HashMap<String, String> = HashMap::new();

    for (sources, target) in waiting_edges {
        // Barrier channel name must match what compile_with() created.
        // sources is a Vec so we preserve insertion order for the name.
        let barrier_name = format!("join:{}:{}", sources.join("+"), target);

        // Each source must write its name into this barrier channel
        for source in sources {
            join_writes_for_source
                .entry(source.clone())
                .or_default()
                .push((barrier_name.clone(), source.clone()));
        }

        // The target node is triggered by the barrier channel
        join_trigger_for_target.insert(target.clone(), barrier_name);
    }

    // Build PregelNode for each registered node
    for (name, spec) in nodes {
        // Determine this node's trigger channel.
        // Join-target nodes use their barrier channel; all others use the
        // standard ephemeral "branch:to:{name}" channel.
        let trigger = join_trigger_for_target
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("branch:to:{}", name));

        // Determine input channels — all non-special channels
        let input_channels: Vec<String> = channels
            .keys()
            .filter(|k| {
                !k.starts_with("branch:") && !k.starts_with("join:") && *k != START
            })
            .cloned()
            .collect();

        // Plain edge targets for this node
        let targets: Vec<String> = edge_targets.get(name).cloned().unwrap_or_default();

        // Barrier channel writes this node must emit when it completes
        // (participates in one or more join edges)
        let barrier_writes: Vec<(String, String)> = join_writes_for_source
            .get(name)
            .cloned()
            .unwrap_or_default();

        // Branch specs
        let node_branches: Vec<BranchSpec> = branches
            .get(name)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default();

        let node_runnable = spec.runnable.clone();
        let node_name = name.clone();

        let combined: Arc<dyn Runnable> = Arc::new(
            crate::runnable::RunnableCallable::new(
                node_name.clone(),
                move |input, config| {
                    let node_runnable = node_runnable.clone();
                    let targets = targets.clone();
                    let barrier_writes = barrier_writes.clone();
                    let node_branches = node_branches.clone();
                    async move {
                        // 1. Execute the node logic
                        let output = node_runnable.ainvoke(&input, &config).await?;

                        // 2. Build combined output: state updates + trigger writes
                        let mut result = serde_json::Map::new();

                        // Copy state updates from node output
                        if let Some(obj) = output.as_object() {
                            for (k, v) in obj {
                                result.insert(k.clone(), v.clone());
                            }
                        }

                        // 3. Write to plain trigger channels for simple edge targets
                        for target in &targets {
                            let trigger_ch = format!("branch:to:{}", target);
                            result.insert(trigger_ch, JsonValue::String(target.clone()));
                        }

                        // 4. Write into barrier channels for join-edge participation.
                        // The value written is this node's own name so the
                        // NamedBarrierValue can track which sources have arrived.
                        for (barrier_ch, source_name) in &barrier_writes {
                            result.insert(
                                barrier_ch.clone(),
                                JsonValue::String(source_name.clone()),
                            );
                        }

                        // 5. Evaluate conditional branches
                        for branch in &node_branches {
                            let branch_result = branch.path.ainvoke(&output, &config).await?;
                            let key = branch_result.as_str().unwrap_or("");
                            if let Some(target) = branch.resolve(key) {
                                let trigger_ch = format!("branch:to:{}", target);
                                result.insert(trigger_ch, JsonValue::String(target));
                            }
                        }

                        Ok(JsonValue::Object(result))
                    }
                },
            ),
        );

        let pregel_node = PregelNode::new(
            input_channels,
            vec![trigger],
            combined,
        );

        pregel_nodes.insert(name.clone(), pregel_node);
    }

    pregel_nodes
}

/// Default recursion limit.
const DEFAULT_RECURSION_LIMIT: u64 = 25;

// ── Streaming context ─────────────────────────────────────────────────────
//
// Passed to `run_pregel_inner` when streaming is enabled. When `None`, the
// inner loop runs in non-streaming mode (same logic, no emit calls).

struct StreamCtx<'a> {
    modes: &'a HashSet<StreamMode>,
    tx: &'a mpsc::Sender<StreamPart>,
    /// Sender for the `Custom` stream channel. `None` when Custom mode is off.
    custom_tx: Option<mpsc::Sender<JsonValue>>,
}

impl<'a> StreamCtx<'a> {
    fn has(&self, mode: &StreamMode) -> bool {
        self.modes.contains(mode)
    }
}

// Helper: apply completed tasks' writes to channels, updating versions_seen
// and channel_versions for them. The interrupted task (identified by
// `interrupted_task_id`) is deliberately excluded so its trigger channels
// remain "unseen" and it re-triggers on resume.
//
// This mirrors Python's `_suppress_interrupt` in `_loop.py`.
fn apply_completed_writes(
    interrupted_task_id: &str,
    tasks: &[PregelExecutableTask],
    channels: &HashMap<String, Box<dyn Channel>>,
    versions_seen: &mut HashMap<String, HashMap<String, JsonValue>>,
    channel_versions: &mut ChannelVersions,
) {
    // Update versions_seen only for completed tasks
    for task in tasks.iter().filter(|t| t.id != interrupted_task_id && !t.writes.is_empty()) {
        let seen = versions_seen.entry(task.name.clone()).or_default();
        for trigger in &task.triggers {
            if let Some(ver) = channel_versions.get(trigger.as_str()) {
                seen.insert(trigger.clone(), ver.clone());
            }
        }
    }

    // Compute a single global next_version from the max of all channel versions
    let max_ver = channel_versions
        .values()
        .filter_map(|v| v.as_str().and_then(|s| s.parse::<u64>().ok()))
        .max()
        .unwrap_or(0);
    let next_version = JsonValue::String(format!("{:032}", max_ver + 1));

    // Collect and apply writes from completed tasks to channels
    let mut writes_by_channel: HashMap<String, Vec<JsonValue>> = HashMap::new();
    for task in tasks.iter().filter(|t| t.id != interrupted_task_id && !t.writes.is_empty()) {
        for (chan, val) in &task.writes {
            if chan != crate::constants::TASKS && chan != crate::constants::INTERRUPT {
                writes_by_channel.entry(chan.clone()).or_default().push(val.clone());
            }
        }
    }

    for (chan, vals) in &writes_by_channel {
        if let Some(ch) = channels.get(chan.as_str()) {
            if ch.update(vals).unwrap_or(false) {
                channel_versions.insert(chan.clone(), next_version.clone());
            }
        }
    }
}

// Helper: collect output channel keys (excluding internal routing channels).
fn output_channel_keys(channels: &HashMap<String, Box<dyn Channel>>) -> Vec<String> {
    channels
        .keys()
        .filter(|k| !k.starts_with("branch:") && !k.starts_with("join:") && *k != START)
        .cloned()
        .collect()
}

// Helper: bump-version closure used in apply_writes.
fn bump_version(current: Option<&JsonValue>) -> JsonValue {
    let num = current
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    JsonValue::String(format!("{:032}", num + 1))
}

impl CompiledStateGraph {
    // ────────────────────────────────────────────────────────────────────────
    // Public thin wrappers
    // ────────────────────────────────────────────────────────────────────────

    /// Non-streaming invocation: runs the BSP loop and returns the final output.
    async fn run_pregel(
        &self,
        input: &JsonValue,
        config: &RunnableConfig,
    ) -> Result<JsonValue, RunnableError> {
        self.run_pregel_inner(input, config, None).await
    }

    /// Streaming invocation: runs the BSP loop and emits `StreamPart`s via `tx`.
    async fn run_pregel_streaming(
        &self,
        input: &JsonValue,
        config: &RunnableConfig,
        modes: &HashSet<StreamMode>,
        tx: &mpsc::Sender<StreamPart>,
    ) -> Result<JsonValue, RunnableError> {
        // Set up the custom-stream forwarder if Custom mode is requested.
        let (custom_tx, has_custom) = if modes.contains(&StreamMode::Custom) {
            let (ctx, mut crx) = mpsc::channel::<JsonValue>(64);
            let tx_clone = tx.clone();
            tokio::spawn(async move {
                while let Some(data) = crx.recv().await {
                    let _ = tx_clone.send(StreamPart::custom(vec![], data)).await;
                }
            });
            (Some(ctx), true)
        } else {
            (None, false)
        };
        let _ = has_custom; // suppresses warning; custom_tx presence implies has_custom

        let ctx = StreamCtx { modes, tx, custom_tx };
        self.run_pregel_inner(input, config, Some(&ctx)).await
    }

    /// Public streaming API: returns a `ReceiverStream` of `StreamPart`s.
    pub fn astream(
        &self,
        input: &JsonValue,
        config: &RunnableConfig,
        stream_modes: Vec<StreamMode>,
    ) -> ReceiverStream<StreamPart> {
        let (tx, rx) = mpsc::channel(256);
        let modes: HashSet<StreamMode> = stream_modes.into_iter().collect();

        let graph = self.clone();
        let input = input.clone();
        let config = config.clone();

        tokio::spawn(async move {
            let result = graph.run_pregel_streaming(&input, &config, &modes, &tx).await;
            if let Err(e) = result {
                let _ = tx.send(StreamPart::debug(
                    vec![],
                    serde_json::json!({"error": e.to_string()}),
                )).await;
            }
        });

        ReceiverStream::new(rx)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Unified BSP loop (previously duplicated as run_pregel / run_pregel_streaming)
    // ────────────────────────────────────────────────────────────────────────
    //
    // `stream` is `None` in non-streaming mode and `Some(&ctx)` in streaming
    // mode. Every streaming emit is guarded by `if let Some(s) = stream`.
    // The core logic — checkpoint loading, task preparation, execution,
    // apply_writes, interrupt handling — is identical in both modes.
    async fn run_pregel_inner(
        &self,
        input: &JsonValue,
        config: &RunnableConfig,
        stream: Option<&StreamCtx<'_>>,
    ) -> Result<JsonValue, RunnableError> {
        let mut config = config.clone();
        // ── Setup ────────────────────────────────────────────────────────────

        let pregel_nodes = build_pregel_nodes(
            &self.nodes,
            &self.edges,
            &self.waiting_edges,
            &self.branches,
            &self.channels,
        );
        let trigger_to_nodes = crate::pregel::build_trigger_to_nodes(&pregel_nodes);

        // Load checkpoint (for resume support)
        let mut saved_checkpoint_exists = false;
        let (mut channels, mut channel_versions, mut versions_seen) =
            if let Some(ref cp) = self.checkpointer {
                match cp.get_tuple(&config) {
                    Ok(Some(tuple)) => {
                        saved_checkpoint_exists = true;
                        let cp_channels: HashMap<String, Option<JsonValue>> = tuple
                            .checkpoint
                            .channel_values
                            .iter()
                            .map(|(k, v)| (k.clone(), Some(v.clone())))
                            .collect();
                        let restored = channels_from_checkpoint(&self.channels, &cp_channels);

                        // Apply non-RESUME pending writes from the checkpoint
                        if let Some(ref pending) = tuple.pending_writes {
                            for (_task_id, channel, value) in pending {
                                if channel != RESUME {
                                    if let Some(ch) = restored.get(channel) {
                                        ch.update(&[value.clone()]).ok();
                                    }
                                }
                            }
                        }

                        (
                            restored,
                            tuple.checkpoint.channel_versions.clone(),
                            tuple.checkpoint.versions_seen.clone(),
                        )
                    }
                    _ => (
                        self.channels.iter().map(|(k, c)| (k.clone(), c.clone_channel())).collect(),
                        HashMap::new(),
                        HashMap::new(),
                    ),
                }
            } else {
                (
                    self.channels.iter().map(|(k, c)| (k.clone(), c.clone_channel())).collect(),
                    HashMap::new(),
                    HashMap::new(),
                )
            };

        // BSP loop counters
        let mut step: u64 = 0;
        let max_steps = config.get_recursion_limit().unwrap_or(self.recursion_limit);
        let mut last_output = JsonValue::Null;
        let mut pending_writes: Vec<(String, String, JsonValue)> = Vec::new();

        // Version offset: ensures new trigger writes have strictly higher
        // versions than anything the checkpoint has already seen.
        let version_offset: u64 = if saved_checkpoint_exists {
            channel_versions
                .values()
                .filter_map(|v| v.as_str().and_then(|s| s.parse::<u64>().ok()))
                .max()
                .unwrap_or(0)
                + 1
        } else {
            0
        };

        // Detect resume-from-Command vs. fresh invocation vs. fork
        let is_resuming = if let Ok(cmd) = serde_json::from_value::<Command>(input.clone()) {
            let cmd_writes = map_command(&cmd);
            let has_resume = cmd_writes.iter().any(|(_, chan, _)| chan == RESUME);
            pending_writes.extend(cmd_writes);
            has_resume
        } else {
            false
        };
        let is_fork = input.is_null() && saved_checkpoint_exists;

        // Write input to channels on a fresh invocation OR when providing new input to a resumed thread
        if !is_fork && (!is_resuming || !input.is_null()) {
            let input_writes = map_input(&[START.to_string()], input);
            for (chan, val) in &input_writes {
                if let Some(ch) = channels.get(chan) {
                    ch.update(&[val.clone()]).ok();
                }
            }
            if let Some(obj) = input.as_object() {
                for (key, val) in obj {
                    if key != START && !key.starts_with("branch:") && !key.starts_with("join:") {
                        if let Some(ch) = channels.get(key) {
                            ch.update(&[val.clone()]).ok();
                        }
                    }
                }
            }
            for (chan, _) in &input_writes {
                channel_versions.insert(
                    chan.clone(),
                    JsonValue::String(format!("{:032}", version_offset + step)),
                );
            }
            // Kick off the first nodes by writing START edge trigger channels
            for (start, end) in &self.edges {
                if start == START && end != END {
                    let trigger_ch = format!("branch:to:{}", end);
                    if let Some(ch) = channels.get(&trigger_ch) {
                        ch.update(&[JsonValue::String(end.clone())]).ok();
                        channel_versions.insert(
                            trigger_ch,
                            JsonValue::String(format!("{:032}", version_offset + step)),
                        );
                    }
                }
            }
        }

        // ── Super-step loop ──────────────────────────────────────────────────

        while step < max_steps {
            let checkpoint_id = format!("{:032}", version_offset + step);

            // PLAN: determine which nodes to run this step
            let mut tasks = prepare_next_tasks(
                &pregel_nodes,
                &channels,
                &config,
                version_offset + step,
                &mut versions_seen,
                &trigger_to_nodes,
                None,
                &checkpoint_id,
                &pending_writes,
                &channel_versions,
            );



            if tasks.is_empty() {
                break;
            }
            
            // Consume pending writes (especially RESUME) so they don't apply to subsequent supersteps
            pending_writes.clear();

            // ── Streaming: emit task-start events ───────────────────────────
            if let Some(s) = stream {
                if s.has(&StreamMode::Tasks) {
                    for task in &tasks {
                        let data = serde_json::json!({
                            "id": task.id,
                            "name": task.name,
                            "triggers": task.triggers,
                        });
                        let _ = s.tx.send(StreamPart::tasks(vec![], data)).await;
                    }
                }
            }

            // interrupt_before: pause before running the matched nodes
            if !self.interrupt_before.is_empty() {
                let task_names: Vec<String> = tasks.iter().map(|t| t.name.clone()).collect();
                if task_names.iter().any(|n| self.interrupt_before.contains(n)) {
                    if let Some(ref cp) = self.checkpointer {
                        if let Some(new_config) = self.save_checkpoint(cp, &config, &channels, &channel_versions, &versions_seen) {
                            config = new_config;
                        }
                    }
                    // Streaming: emit values before returning
                    if let Some(s) = stream {
                        if s.has(&StreamMode::Values) {
                            let keys = output_channel_keys(&channels);
                            let _ = s.tx.send(StreamPart::values(vec![], read_channels(&channels, &keys))).await;
                        }
                    }
                    let keys = output_channel_keys(&channels);
                    return Ok(read_channels(&channels, &keys));
                }
            }

            // EXECUTE: build runner (with custom-stream writer in streaming mode)
            let runner = if let Some(s) = stream {
                let runtime = Arc::new(crate::runtime::Runtime {
                    context: (),
                    store: self.store.clone(),
                    stream_writer: s.custom_tx.clone(),
                    previous: None,
                    execution_info: None,
                    server_info: None,
                });
                if s.custom_tx.is_some() {
                    PregelRunner::new(Some(runtime.clone()))
                        .with_stream_writer(s.custom_tx.clone().unwrap())
                } else {
                    PregelRunner::new(Some(runtime))
                }
            } else {
                PregelRunner::new(self.store.clone().map(|_| {
                    Arc::new(crate::runtime::Runtime {
                        context: (),
                        store: self.store.clone(),
                        stream_writer: None,
                        previous: None,
                        execution_info: None,
                        server_info: None,
                    })
                }))
            };

            match runner.run_tasks(&mut tasks).await {
                Ok(()) => {}

                Err(crate::pregel::runner::RunnerError::Interrupt { task_id, interrupt }) => {
                    // Mirrors Python's _suppress_interrupt:
                    // Apply writes from tasks that completed *before* the interrupt
                    // so their trigger-channel writes survive into the checkpoint.
                    // The interrupted task is excluded so it re-triggers on resume.
                    apply_completed_writes(
                        &task_id,
                        &tasks,
                        &channels,
                        &mut versions_seen,
                        &mut channel_versions,
                    );

                    // Save checkpoint (now includes completed tasks' channel writes)
                    if let Some(ref cp) = self.checkpointer {
                        if let Some(new_config) = self.save_checkpoint(cp, &config, &channels, &channel_versions, &versions_seen) {
                            config = new_config;
                        }
                        // Save interrupt as pending writes for get_state()
                        let iw: Vec<(String, String, JsonValue)> = interrupt
                            .interrupts
                            .iter()
                            .map(|iv| {
                                let val = serde_json::to_value(iv).unwrap_or(JsonValue::Null);
                                (task_id.clone(), crate::constants::INTERRUPT.to_string(), val)
                            })
                            .collect();
                        if !iw.is_empty() {
                            if let Err(e) = cp.put_writes(&config, &iw, &task_id, "") {
                                eprintln!("[CHECKPOINT] Failed to save interrupt writes: {}", e);
                            }
                        }
                    }

                    // Streaming: emit values before returning
                    if let Some(s) = stream {
                        if s.has(&StreamMode::Values) {
                            let keys = output_channel_keys(&channels);
                            let _ = s.tx.send(StreamPart::values(vec![], read_channels(&channels, &keys))).await;
                        }
                    }

                    let keys = output_channel_keys(&channels);
                    return Ok(read_channels(&channels, &keys));
                }

                Err(other) => return Err(RunnableError::Runner(other.to_string())),
            }

            // ── Streaming: emit per-node updates ────────────────────────────
            if let Some(s) = stream {
                if s.has(&StreamMode::Updates) {
                    for task in &tasks {
                        if !task.writes.is_empty() {
                            let mut node_updates = serde_json::Map::new();
                            for (chan, val) in &task.writes {
                                if !chan.starts_with("branch:") && !chan.starts_with("join:") {
                                    node_updates.insert(chan.clone(), val.clone());
                                }
                            }
                            if !node_updates.is_empty() {
                                let data = serde_json::json!({ &task.name: node_updates });
                                let _ = s.tx.send(StreamPart::updates(vec![], data)).await;
                            }
                        }
                    }
                }
            }

            // UPDATE: apply all task writes to channels
            apply_writes(
                &mut channels,
                &tasks,
                &mut versions_seen,
                &mut channel_versions,
                &trigger_to_nodes,
                bump_version,
            );

            // Save "loop" checkpoint after each completed super-step
            if let Some(ref cp) = self.checkpointer {
                if let Some(new_config) = self.save_checkpoint(cp, &config, &channels, &channel_versions, &versions_seen) {
                    config = new_config;
                }
            }

            // ── Streaming: emit values after writes ──────────────────────────
            if let Some(s) = stream {
                if s.has(&StreamMode::Values) {
                    let keys = output_channel_keys(&channels);
                    let _ = s.tx.send(StreamPart::values(vec![], read_channels(&channels, &keys))).await;
                }
            }

            // Read output
            let keys = output_channel_keys(&channels);
            let output = read_channels(&channels, &keys);
            if !output.is_null() {
                last_output = output;
            }

            // interrupt_after: pause after the matched nodes complete
            if !self.interrupt_after.is_empty() {
                let task_names: Vec<String> = tasks.iter().map(|t| t.name.clone()).collect();
                if task_names.iter().any(|n| self.interrupt_after.contains(n)) {
                    return Ok(last_output);
                }
            }

            step += 1;
        }

        Ok(last_output)
    }


}

#[async_trait]
impl Runnable for CompiledStateGraph {
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        // Block on the async implementation
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.run_pregel(input, config)),
            Err(_) => tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(self.run_pregel(input, config)),
        }
    }

    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        self.run_pregel(input, config).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::LastValue;
    use serde_json::json;

    fn make_channels() -> HashMap<String, Box<dyn Channel>> {
        let mut channels = HashMap::new();
        channels.insert("value".to_string(), Box::new(LastValue::new("value")) as Box<dyn Channel>);
        channels
    }

    #[tokio::test]
    async fn test_simple_linear_graph() {
        let mut graph = StateGraph::new(make_channels());

        graph
            .add_node("a", |_input, _config| async { Ok(json!({"value": 1})) })
            .unwrap();
        graph
            .add_node("b", |_input, _config| async { Ok(json!({"value": 2})) })
            .unwrap();

        graph.add_edge(START, "a").unwrap();
        graph.add_edge("a", "b").unwrap();
        graph.add_edge("b", END).unwrap();

        let compiled = graph.compile().unwrap();
        assert!(compiled.has_node("a"));
        assert!(compiled.has_node("b"));
        assert_eq!(compiled.node_names().len(), 2);
    }

    #[test]
    fn test_duplicate_node_error() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("a", |_input, _config| async { Ok(json!({})) }).unwrap();
        let result = graph.add_node("a", |_input, _config| async { Ok(json!({})) });
        assert!(result.is_err());
    }

    #[test]
    fn test_reserved_name_error() {
        let mut graph = StateGraph::new(make_channels());
        let result = graph.add_node(START, |_input, _config| async { Ok(json!({})) });
        assert!(result.is_err());
    }

    #[test]
    fn test_end_as_source_error() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("a", |_input, _config| async { Ok(json!({})) }).unwrap();
        let result = graph.add_edge(END, "a");
        assert!(result.is_err());
    }

    #[test]
    fn test_start_as_target_error() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("a", |_input, _config| async { Ok(json!({})) }).unwrap();
        let result = graph.add_edge("a", START);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_start_edge_error() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("a", |_input, _config| async { Ok(json!({})) }).unwrap();
        let result = graph.compile();
        assert!(result.is_err());
    }

    #[test]
    fn test_join_edge() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("a", |_input, _config| async { Ok(json!({})) }).unwrap();
        graph.add_node("b", |_input, _config| async { Ok(json!({})) }).unwrap();
        graph.add_node("c", |_input, _config| async { Ok(json!({})) }).unwrap();

        graph.add_edge(START, "a").unwrap();
        graph.add_edge(START, "b").unwrap();
        graph.add_join_edge(vec!["a".to_string(), "b".to_string()], "c").unwrap();
        graph.add_edge("c", END).unwrap();

        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.node_names().len(), 3);
    }

    #[test]
    fn test_conditional_edges() {
        let mut graph = StateGraph::new(make_channels());
        graph.add_node("agent", |_input, _config| async { Ok(json!({})) }).unwrap();
        graph.add_node("tools", |_input, _config| async { Ok(json!({})) }).unwrap();

        graph.add_edge(START, "agent").unwrap();
        graph
            .add_conditional_edges(
                "agent",
                |_input, _config| async { Ok(json!("continue")) },
                Some(HashMap::from([
                    ("continue".to_string(), "tools".to_string()),
                    ("end".to_string(), END.to_string()),
                ])),
            )
            .unwrap();
        graph.add_edge("tools", "agent").unwrap();

        let compiled = graph.compile().unwrap();
        assert!(compiled.has_node("agent"));
        assert!(compiled.has_node("tools"));
    }

    #[tokio::test]
    async fn test_invoke_linear_graph() {
        // End-to-end test: build graph → compile → invoke → check output
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert("count".to_string(), Box::new(LastValue::new("count")) as Box<dyn Channel>);

        let mut graph = StateGraph::new(channels);

        graph
            .add_node("increment", |_input, _config| async {
                Ok(json!({"count": 1}))
            })
            .unwrap();
        graph
            .add_node("double", |_input, _config| async {
                Ok(json!({"count": 2}))
            })
            .unwrap();

        graph.add_edge(START, "increment").unwrap();
        graph.add_edge("increment", "double").unwrap();
        graph.add_edge("double", END).unwrap();

        let compiled = graph.compile().unwrap();
        let config = RunnableConfig::new();
        let result = compiled.ainvoke(&json!({"count": 0}), &config).await.unwrap();

        // The output should contain the "count" channel value
        assert!(result.is_object());
        // After "double" runs, count should be 2
        assert_eq!(result.get("count"), Some(&json!(2)));
    }

    #[tokio::test]
    async fn test_invoke_single_node() {
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert("result".to_string(), Box::new(LastValue::new("result")) as Box<dyn Channel>);

        let mut graph = StateGraph::new(channels);
        graph
            .add_node("process", |_input, _config| async {
                Ok(json!({"result": 42}))
            })
            .unwrap();
        graph.add_edge(START, "process").unwrap();
        graph.add_edge("process", END).unwrap();

        let compiled = graph.compile().unwrap();
        let config = RunnableConfig::new();
        let result = compiled.ainvoke(&json!({}), &config).await.unwrap();

        assert_eq!(result.get("result"), Some(&json!(42)));
    }

    #[tokio::test]
    async fn test_interrupt_before() {
        // Test interrupt_before: graph pauses before executing the specified node
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert("value".to_string(), Box::new(LastValue::new("value")) as Box<dyn Channel>);

        let mut graph = StateGraph::new(channels);

        graph
            .add_node("process", |_input, _config| async {
                Ok(json!({"value": 42}))
            })
            .unwrap();
        graph.add_edge(START, "process").unwrap();
        graph.add_edge("process", END).unwrap();

        let mut compiled = graph.compile().unwrap();
        // Set interrupt_before to pause before "process" node
        compiled.interrupt_before = vec!["process".to_string()];

        let config = RunnableConfig::new();
        let result = compiled.ainvoke(&json!({}), &config).await.unwrap();

        // Graph should return current state (empty since process hasn't run yet)
        assert!(result.is_object());
        // The "value" channel should not have been set yet
        assert!(result.get("value").is_none() || result.get("value").unwrap().is_null());
    }

    #[tokio::test]
    async fn test_interrupt_after() {
        // Test interrupt_after: graph pauses after executing the specified node
        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert("value".to_string(), Box::new(LastValue::new("value")) as Box<dyn Channel>);

        let mut graph = StateGraph::new(channels);

        graph
            .add_node("process", |_input, _config| async {
                Ok(json!({"value": 42}))
            })
            .unwrap();
        graph.add_edge(START, "process").unwrap();
        graph.add_edge("process", END).unwrap();

        let mut compiled = graph.compile().unwrap();
        // Set interrupt_after to pause after "process" node
        compiled.interrupt_after = vec!["process".to_string()];

        let config = RunnableConfig::new();
        let result = compiled.ainvoke(&json!({}), &config).await.unwrap();

        // Graph should return current state with the value from "process"
        assert!(result.is_object());
        assert_eq!(result.get("value"), Some(&json!(42)));
    }

    #[tokio::test]
    async fn test_update_state() {
        use crate::channels::LastValue;
        use langgraph_checkpoint::checkpoint::memory::InMemorySaver;

        let mut channels: HashMap<String, Box<dyn Channel>> = HashMap::new();
        channels.insert("name".to_string(), Box::new(LastValue::new("name")) as Box<dyn Channel>);
        channels.insert("value".to_string(), Box::new(LastValue::new("value")) as Box<dyn Channel>);

        let mut graph = StateGraph::new(channels);
        graph
            .add_node("set_value", |_input, _config| async {
                Ok(json!({"value": 42}))
            })
            .unwrap();
        graph.add_edge(START, "set_value").unwrap();
        graph.add_edge("set_value", END).unwrap();

        let checkpointer = Arc::new(InMemorySaver::new());
        let compiled = graph.compile_builder()
            .checkpointer(checkpointer)
            .build()
            .unwrap();

        let mut config = RunnableConfig::new();
        config.insert("configurable".to_string(), json!({"thread_id": "test-thread"}));

        // First invoke
        let result = compiled.ainvoke(&json!({"name": "original"}), &config).await.unwrap();
        assert_eq!(result.get("value"), Some(&json!(42)));

        // Verify get_state
        let snapshot = compiled.get_state(&config).unwrap();
        assert_eq!(snapshot.values.get("name").and_then(|v| v.as_str()), Some("original"));
        assert_eq!(snapshot.values.get("value").and_then(|v| v.as_i64()), Some(42));

        // Update state
        compiled.update_state(&config, &json!({"name": "updated"})).unwrap();

        // Verify update took effect
        let snapshot = compiled.get_state(&config).unwrap();
        assert_eq!(snapshot.values.get("name").and_then(|v| v.as_str()), Some("updated"));
        assert_eq!(snapshot.values.get("value").and_then(|v| v.as_i64()), Some(42));
    }
}
