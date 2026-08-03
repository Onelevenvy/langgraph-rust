use super::types::*;
use crate::config::RunnableConfig;
use crate::error::CheckpointError;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Default WRITES_IDX_MAP for special write channels
pub fn writes_idx_map() -> HashMap<&'static str, i64> {
    let mut m = HashMap::new();
    m.insert("__error__", -1i64);
    m.insert("__scheduled__", -2i64);
    m.insert("__interrupt__", -3i64);
    m.insert("__resume__", -4i64);
    m
}

/// Metadata keys excluded from checkpoint metadata
pub fn excluded_metadata_keys() -> &'static [&'static str] {
    &[
        "thread_id",
        "checkpoint_id",
        "checkpoint_ns",
        "checkpoint_map",
        "langgraph_step",
        "langgraph_node",
        "langgraph_triggers",
        "langgraph_path",
        "langgraph_checkpoint_ns",
    ]
}

/// Base checkpoint saver trait. Mirrors Python's BaseCheckpointSaver.
///
/// All methods that store/retrieve checkpoints must be implemented.
/// Async versions default to wrapping sync versions via spawn_blocking.
#[async_trait]
pub trait BaseCheckpointSaver: Send + Sync {
    /// Get a checkpoint tuple by config.
    fn get_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError>;

    /// List checkpoint tuples.
    fn list(
        &self,
        config: Option<&RunnableConfig>,
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError>;

    /// Store a checkpoint.
    ///
    /// Takes ownership of the `Checkpoint` so savers can move the serialized
    /// state instead of deep-copying it on the hot per-super-step path.
    fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError>;

    /// Store pending writes for a checkpoint.
    fn put_writes(
        &self,
        config: &RunnableConfig,
        writes: &[(String, String, JsonValue)],
        task_id: &str,
        task_path: &str,
    ) -> Result<(), CheckpointError>;

    /// Delete all checkpoints for a thread.
    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError>;

    /// Get the next version for a channel.
    fn get_next_version(&self, current: Option<&ChannelVersion>) -> ChannelVersion {
        match current {
            Some(JsonValue::Number(n)) => {
                let v = n.as_i64().unwrap_or(0) + 1;
                JsonValue::Number(v.into())
            }
            Some(JsonValue::String(s)) => {
                // Parse "NNN.random" format
                let num: i64 = s.split('.').next().unwrap_or("0").parse().unwrap_or(0);
                JsonValue::String(format!("{:032}.{:016}", num + 1, rand::random::<u64>()))
            }
            _ => JsonValue::Number(1i64.into()),
        }
    }

    // Async mirrors with default implementations
    //
    // The defaults bridge to the sync methods. `block_in_place` is only valid
    // on a multi-thread runtime (it panics on `current_thread`), so when no
    // multi-thread runtime is present the sync method is called directly —
    // the same behavior a synchronous caller gets today.

    async fn aget_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread {
                let config = config.clone();
                let this = self;
                return tokio::task::block_in_place(|| this.get_tuple(&config));
            }
        }
        self.get_tuple(config)
    }

    async fn aput(
        &self,
        config: &RunnableConfig,
        checkpoint: Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread {
                let config = config.clone();
                return tokio::task::block_in_place(move || {
                    self.put(&config, checkpoint, metadata, new_versions)
                });
            }
        }
        self.put(config, checkpoint, metadata, new_versions)
    }

    async fn aput_writes(
        &self,
        config: &RunnableConfig,
        writes: Vec<(String, String, JsonValue)>,
        task_id: String,
        task_path: String,
    ) -> Result<(), CheckpointError> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread {
                let config = config.clone();
                return tokio::task::block_in_place(|| {
                    self.put_writes(&config, &writes, &task_id, &task_path)
                });
            }
        }
        self.put_writes(config, &writes, &task_id, &task_path)
    }

    async fn adelete_thread(&self, thread_id: String) -> Result<(), CheckpointError> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            if handle.runtime_flavor() != tokio::runtime::RuntimeFlavor::CurrentThread {
                let this = self;
                return tokio::task::block_in_place(|| this.delete_thread(&thread_id));
            }
        }
        self.delete_thread(&thread_id)
    }
}

/// Helper to extract checkpoint_id from config
pub fn get_checkpoint_id(config: &RunnableConfig) -> Option<String> {
    config
        .get("configurable")
        .and_then(|c| c.get("checkpoint_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Helper to extract checkpoint metadata from config
pub fn get_checkpoint_metadata(
    config: &RunnableConfig,
    metadata: &CheckpointMetadata,
) -> CheckpointMetadata {
    let mut meta = metadata.clone();
    if let Some(step) = config
        .get("configurable")
        .and_then(|c| c.get("langgraph_step"))
        .and_then(|v| v.as_i64())
    {
        meta.step = Some(step);
    }
    meta
}

/// Copy a checkpoint
pub fn copy_checkpoint(checkpoint: &Checkpoint) -> Checkpoint {
    checkpoint.copy()
}

/// Create an empty checkpoint
pub fn empty_checkpoint() -> Checkpoint {
    Checkpoint::empty()
}

/// Create a checkpoint from current channel state
pub fn create_checkpoint(
    checkpoint: &Checkpoint,
    channel_values: HashMap<String, JsonValue>,
    _step: i64,
) -> Checkpoint {
    use crate::checkpoint::id::uuid6;
    use chrono::Utc;

    Checkpoint {
        v: LATEST_VERSION,
        id: uuid6(),
        ts: Utc::now().to_rfc3339(),
        channel_values,
        channel_versions: checkpoint.channel_versions.clone(),
        versions_seen: checkpoint.versions_seen.clone(),
        updated_channels: checkpoint.updated_channels.clone(),
    }
}

// Add rand dependency for version generation
mod rand {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    pub fn random<T: From<u64>>() -> T {
        let s = RandomState::new();
        let mut hasher = s.build_hasher();
        hasher.write_u64(42); // Fixed seed for determinism in tests
        T::from(hasher.finish())
    }
}
