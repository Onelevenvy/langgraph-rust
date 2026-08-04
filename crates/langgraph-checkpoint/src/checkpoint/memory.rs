use super::base::BaseCheckpointSaver;
use super::types::*;
use crate::config::RunnableConfig;
use crate::error::CheckpointError;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

type RowKey = (String, String, String); // (thread_id, checkpoint_ns, checkpoint_id)
type WriteKey = (String, String, String, i64); // (thread_id, checkpoint_ns, checkpoint_id, idx)
/// Blob key: (thread_id, checkpoint_ns, channel, version_str)
type BlobKey = (String, String, String, String);

/// (thread_id, checkpoint_ns, checkpoint_id) -> (checkpoint, metadata, parent_checkpoint_id)
type RowValue = (Checkpoint, CheckpointMetadata, Option<String>);
/// (thread_id, checkpoint_ns, checkpoint_id, idx) -> (task_id, channel, value_json, task_path)
type WriteValue = (String, String, JsonValue, String);

/// In-memory checkpoint saver for testing and development.
///
/// Stores `Checkpoint`s directly (no JSON round trip). Channel values are
/// stored as version-addressed blobs — `put` receives a delta
/// (`new_versions` only, see `BaseCheckpointSaver`), and reads reconstruct
/// the full state by resolving each channel's version against the blob
/// store, mirroring the version-joined merge of the sqlite saver. `None`
/// marks a channel whose version moved but has no value (e.g. a cleared
/// ephemeral channel), matching the "empty" blob rows of the sqlite saver.
/// The newest checkpoint per thread is tracked O(1).
pub struct InMemorySaver {
    /// Checkpoint rows with `channel_values` stripped (they live in `blobs`).
    rows: RwLock<HashMap<RowKey, RowValue>>,
    /// (thread_id, checkpoint_ns, channel, version) -> value or empty marker.
    blobs: RwLock<HashMap<BlobKey, Option<JsonValue>>>,
    /// (thread_id, checkpoint_ns) -> newest checkpoint_id
    latest: RwLock<HashMap<(String, String), String>>,
    writes: RwLock<HashMap<WriteKey, WriteValue>>,
}

impl InMemorySaver {
    pub fn new() -> Self {
        Self {
            rows: RwLock::new(HashMap::new()),
            blobs: RwLock::new(HashMap::new()),
            latest: RwLock::new(HashMap::new()),
            writes: RwLock::new(HashMap::new()),
        }
    }

    /// Reconstruct a checkpoint's full `channel_values` from the
    /// version-addressed blob store: every channel in `channel_versions`
    /// resolves to the blob written at that version (possibly by an earlier
    /// checkpoint), empty markers and missing blobs are skipped.
    fn reconstruct_values(
        blobs: &HashMap<BlobKey, Option<JsonValue>>,
        channel_versions: &ChannelVersions,
        thread_id: &str,
        checkpoint_ns: &str,
    ) -> HashMap<String, JsonValue> {
        let mut values = HashMap::new();
        for (channel, ver) in channel_versions {
            let ver_str = match ver {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => n.to_string(),
                _ => continue,
            };
            let key = (
                thread_id.to_string(),
                checkpoint_ns.to_string(),
                channel.clone(),
                ver_str,
            );
            if let Some(Some(val)) = blobs.get(&key) {
                values.insert(channel.clone(), val.clone());
            }
        }
        values
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

impl Default for InMemorySaver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseCheckpointSaver for InMemorySaver {
    fn get_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::config_to_ids(config);

        // Resolve the requested checkpoint: explicit id, else the thread's
        // newest (tracked O(1) instead of scanning the whole history).
        let resolved_cid = match checkpoint_id {
            Some(cid) => cid,
            None => match self
                .latest
                .read()
                .get(&(thread_id.clone(), checkpoint_ns.clone()))
            {
                Some(cid) => cid.clone(),
                None => return Ok(None),
            },
        };

        let (mut checkpoint, metadata, parent_cid) = match self.rows.read().get(&(
            thread_id.clone(),
            checkpoint_ns.clone(),
            resolved_cid.clone(),
        )) {
            Some(v) => v.clone(),
            None => return Ok(None),
        };

        // Reconstruct the full state: the row holds only the delta, so merge
        // version-addressed blob values over it (see struct docs).
        let blob_values = {
            let blobs = self.blobs.read();
            Self::reconstruct_values(
                &blobs,
                &checkpoint.channel_versions,
                &thread_id,
                &checkpoint_ns,
            )
        };
        checkpoint.channel_values = blob_values;

        let parent_config = parent_cid.map(|pid| {
            let mut c = RunnableConfig::new();
            c.insert(
                "configurable".to_string(),
                serde_json::json!({
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": pid,
                }),
            );
            c
        });

        // Get pending writes
        let writes = self.writes.read();
        let pending_writes: Vec<PendingWrite> = writes
            .iter()
            .filter(|((tid, ns, cid, _), _)| {
                tid == &thread_id && ns == &checkpoint_ns && cid == &resolved_cid
            })
            .map(|(_, (task_id, channel, value, _))| {
                (task_id.clone(), channel.clone(), value.clone())
            })
            .collect();

        Ok(Some(CheckpointTuple {
            config: {
                let mut c = RunnableConfig::new();
                c.insert(
                    "configurable".to_string(),
                    serde_json::json!({
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": resolved_cid,
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
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        let rows = self.rows.read();

        let (thread_id, checkpoint_ns) = match config {
            Some(c) => {
                let (tid, ns, _) = Self::config_to_ids(c);
                (tid, ns)
            }
            None => (String::new(), String::new()),
        };

        let before_id = before.and_then(|c| Self::config_to_ids(c).2);

        let mut entries: Vec<_> = rows
            .iter()
            .filter(|((tid, ns, _), _)| {
                (thread_id.is_empty() || tid == &thread_id)
                    && (checkpoint_ns.is_empty() || ns == &checkpoint_ns)
            })
            .filter(|((_, _, cid), _)| {
                if let Some(ref bid) = before_id {
                    cid < bid
                } else {
                    true
                }
            })
            .collect();

        // Sort by checkpoint_id descending (most recent first)
        entries.sort_by(|a, b| b.0 .2.cmp(&a.0 .2));

        if let Some(limit) = limit {
            entries.truncate(limit);
        }

        let mut results = Vec::new();
        for ((tid, ns, cid), (checkpoint, metadata, parent_cid)) in entries {
            // Apply filter
            if let Some(filter) = filter {
                let metadata_val: JsonValue = serde_json::to_value(metadata)
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                let mut matches = true;
                for (k, v) in filter {
                    if metadata_val.get(k) != Some(v) {
                        matches = false;
                        break;
                    }
                }
                if !matches {
                    continue;
                }
            }

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

            // Reconstruct the full state from version-addressed blobs.
            let mut checkpoint = checkpoint.clone();
            let blob_values = {
                let blobs = self.blobs.read();
                Self::reconstruct_values(&blobs, &checkpoint.channel_versions, tid, ns)
            };
            checkpoint.channel_values = blob_values;

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
                metadata: metadata.clone(),
                parent_config,
                pending_writes: None,
            });
        }

        Ok(results)
    }

    fn put(
        &self,
        config: &RunnableConfig,
        mut checkpoint: Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        let (thread_id, checkpoint_ns, _) = Self::config_to_ids(config);
        let cid = checkpoint.id.clone();

        // Parent = the thread's current newest checkpoint (O(1)).
        let parent_id = self
            .latest
            .read()
            .get(&(thread_id.clone(), checkpoint_ns.clone()))
            .cloned();

        // Store the delta as version-addressed blobs: one per moved channel,
        // with the value if present or an empty marker (`None`) if the
        // channel was cleared — mirroring the "empty" blob rows of the
        // sqlite saver. The row keeps only the metadata fields; reads
        // reconstruct the full state via `reconstruct_values`.
        let mut channel_values = std::mem::take(&mut checkpoint.channel_values);
        {
            let mut blobs = self.blobs.write();
            for (channel, ver) in new_versions {
                let ver_str = match ver {
                    JsonValue::String(s) => s.clone(),
                    JsonValue::Number(n) => n.to_string(),
                    _ => continue,
                };
                let key = (
                    thread_id.clone(),
                    checkpoint_ns.clone(),
                    channel.clone(),
                    ver_str,
                );
                blobs.insert(key, channel_values.remove(channel));
            }
        }

        let key = (thread_id.clone(), checkpoint_ns.clone(), cid.clone());
        self.rows
            .write()
            .insert(key, (checkpoint, metadata.clone(), parent_id));

        // Track the newest checkpoint per thread. Keep the string-max
        // semantics of the previous whole-history scan: only update when the
        // new id sorts higher.
        let mut latest = self.latest.write();
        latest
            .entry((thread_id.clone(), checkpoint_ns.clone()))
            .and_modify(|existing| {
                if cid > *existing {
                    *existing = cid.clone();
                }
            })
            .or_insert_with(|| cid.clone());

        let mut new_config = RunnableConfig::new();
        new_config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": cid,
            }),
        );
        Ok(new_config)
    }

    fn put_writes(
        &self,
        config: &RunnableConfig,
        writes: &[(String, String, JsonValue)],
        task_id: &str,
        task_path: &str,
    ) -> Result<(), CheckpointError> {
        let (thread_id, checkpoint_ns, checkpoint_id) = Self::config_to_ids(config);
        let checkpoint_id = checkpoint_id.unwrap_or_default();

        let mut writes_map = self.writes.write();
        for (idx, write_tuple) in writes.iter().enumerate() {
            let key = (
                thread_id.clone(),
                checkpoint_ns.clone(),
                checkpoint_id.clone(),
                idx as i64,
            );
            // write_tuple is (String, String, JsonValue) - (task_id, channel, value)
            writes_map.insert(
                key,
                (
                    task_id.to_string(),
                    write_tuple.1.clone(),
                    write_tuple.2.clone(),
                    task_path.to_string(),
                ),
            );
        }
        Ok(())
    }

    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        self.rows.write().retain(|(tid, _, _), _| tid != thread_id);
        self.blobs
            .write()
            .retain(|(tid, _, _, _), _| tid != thread_id);
        self.latest.write().retain(|(tid, _), _| tid != thread_id);
        self.writes
            .write()
            .retain(|(tid, _, _, _), _| tid != thread_id);
        Ok(())
    }

    // Async overrides: this saver is pure in-memory, so the async mirrors are
    // just the sync bodies without the trait's default block_in_place bridge
    // (which requires a multi-thread runtime and adds a thread handoff per
    // call on the hot checkpoint path).

    async fn aget_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        self.get_tuple(config)
    }

    async fn aput(
        &self,
        config: &RunnableConfig,
        checkpoint: Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        self.put(config, checkpoint, metadata, new_versions)
    }

    async fn aput_writes(
        &self,
        config: &RunnableConfig,
        writes: Vec<(String, String, JsonValue)>,
        task_id: String,
        task_path: String,
    ) -> Result<(), CheckpointError> {
        self.put_writes(config, &writes, &task_id, &task_path)
    }

    async fn adelete_thread(&self, thread_id: String) -> Result<(), CheckpointError> {
        self.delete_thread(&thread_id)
    }

    fn get_next_version(&self, current: Option<&ChannelVersion>) -> ChannelVersion {
        match current {
            Some(JsonValue::String(s)) => {
                let num: i64 = s.split('.').next().unwrap_or("0").parse().unwrap_or(0);
                JsonValue::String(format!("{:032}.{:016}", num + 1, random_u64()))
            }
            Some(JsonValue::Number(n)) => JsonValue::Number((n.as_i64().unwrap_or(0) + 1).into()),
            _ => JsonValue::String(format!("{:032}.{:016}", 1, random_u64())),
        }
    }
}

fn random_u64() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut hasher = s.build_hasher();
    hasher.write_u64(std::process::id() as u64);
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_saver() {
        let saver = InMemorySaver::new();
        let config = RunnableConfig::new();
        let result = saver.get_tuple(&config).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_put_and_get() {
        let saver = InMemorySaver::new();
        let checkpoint = Checkpoint::empty();
        let metadata = CheckpointMetadata::default();

        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }),
        );

        let new_config = saver
            .put(&config, checkpoint.clone(), &metadata, &HashMap::new())
            .unwrap();
        let tuple = saver.get_tuple(&new_config).unwrap();
        assert!(tuple.is_some());
        let tuple = tuple.unwrap();
        assert_eq!(tuple.checkpoint.id, checkpoint.id);
    }

    #[test]
    fn test_list_checkpoints() {
        let saver = InMemorySaver::new();

        for i in 0..3 {
            let mut checkpoint = Checkpoint::empty();
            checkpoint.id = format!("cp-{:03}", i);
            let metadata = CheckpointMetadata {
                step: Some(i),
                ..Default::default()
            };

            let mut config = RunnableConfig::new();
            config.insert(
                "configurable".to_string(),
                serde_json::json!({
                    "thread_id": "test-thread",
                    "checkpoint_ns": "",
                }),
            );

            saver
                .put(&config, checkpoint, &metadata, &HashMap::new())
                .unwrap();
        }

        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
            }),
        );

        let results = saver.list(Some(&config), None, None, None).unwrap();
        assert_eq!(results.len(), 3);
        // Should be sorted by checkpoint_id descending
        assert_eq!(results[0].checkpoint.id, "cp-002");
        assert_eq!(results[1].checkpoint.id, "cp-001");
        assert_eq!(results[2].checkpoint.id, "cp-000");
    }

    #[test]
    fn test_delete_thread() {
        let saver = InMemorySaver::new();
        let checkpoint = Checkpoint::empty();
        let metadata = CheckpointMetadata::default();

        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
            }),
        );

        saver
            .put(&config, checkpoint, &metadata, &HashMap::new())
            .unwrap();
        saver.delete_thread("test-thread").unwrap();

        let result = saver.get_tuple(&config).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_put_writes() {
        let saver = InMemorySaver::new();
        let checkpoint = Checkpoint::empty();
        let metadata = CheckpointMetadata::default();

        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
            }),
        );

        let new_config = saver
            .put(&config, checkpoint, &metadata, &HashMap::new())
            .unwrap();

        let writes = vec![
            (
                "channel1".to_string(),
                "write-1".to_string(), // 添加这个缺失的 ID 字段
                JsonValue::String("value1".to_string()),
            ),
            (
                "channel2".to_string(),
                "write-2".to_string(), // 添加这个缺失的 ID 字段
                serde_json::json!(42),
            ),
        ];
        saver
            .put_writes(&new_config, &writes, "task-1", "")
            .unwrap();

        let tuple = saver.get_tuple(&new_config).unwrap().unwrap();
        assert!(tuple.pending_writes.is_some());
        assert_eq!(tuple.pending_writes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_incremental_blob_merge() {
        let saver = InMemorySaver::new();
        let metadata = CheckpointMetadata::default();

        let mut config = RunnableConfig::new();
        config.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
                "checkpoint_ns": "",
            }),
        );

        // cp1: a=1, b=1, both at version 1.
        let mut cp1 = Checkpoint::empty();
        cp1.id = "cp-001".to_string();
        cp1.channel_versions
            .insert("a".into(), serde_json::json!(1));
        cp1.channel_versions
            .insert("b".into(), serde_json::json!(1));
        cp1.channel_values.insert("a".into(), serde_json::json!(1));
        cp1.channel_values.insert("b".into(), serde_json::json!(1));
        let mut new_versions = HashMap::new();
        new_versions.insert("a".into(), serde_json::json!(1));
        new_versions.insert("b".into(), serde_json::json!(1));
        saver.put(&config, cp1, &metadata, &new_versions).unwrap();

        // cp2: only a moves to version 2; b is unchanged.
        let mut cp2 = Checkpoint::empty();
        cp2.id = "cp-002".to_string();
        cp2.channel_versions
            .insert("a".into(), serde_json::json!(2));
        cp2.channel_versions
            .insert("b".into(), serde_json::json!(1));
        cp2.channel_values.insert("a".into(), serde_json::json!(2));
        let mut new_versions = HashMap::new();
        new_versions.insert("a".into(), serde_json::json!(2));
        saver.put(&config, cp2, &metadata, &new_versions).unwrap();

        // Reading cp2 merges b from the earlier checkpoint.
        let mut cfg2 = config.clone();
        cfg2.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-002",
            }),
        );
        let tuple = saver.get_tuple(&cfg2).unwrap().unwrap();
        assert_eq!(
            tuple.checkpoint.channel_values.get("a"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(
            tuple.checkpoint.channel_values.get("b"),
            Some(&serde_json::json!(1))
        );

        // Reading cp1 is unaffected by the later delta.
        let mut cfg1 = config.clone();
        cfg1.insert(
            "configurable".to_string(),
            serde_json::json!({
                "thread_id": "test-thread",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-001",
            }),
        );
        let tuple = saver.get_tuple(&cfg1).unwrap().unwrap();
        assert_eq!(
            tuple.checkpoint.channel_values.get("a"),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            tuple.checkpoint.channel_values.get("b"),
            Some(&serde_json::json!(1))
        );

        // Latest (no explicit checkpoint_id) resolves to cp2 with full state.
        let tuple = saver.get_tuple(&config).unwrap().unwrap();
        assert_eq!(tuple.checkpoint.id, "cp-002");
        assert_eq!(
            tuple.checkpoint.channel_values.get("a"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(
            tuple.checkpoint.channel_values.get("b"),
            Some(&serde_json::json!(1))
        );

        // cp3: a moves to version 3 with no value (cleared channel) -> the
        // empty marker removes it; b stays.
        let mut cp3 = Checkpoint::empty();
        cp3.id = "cp-003".to_string();
        cp3.channel_versions
            .insert("a".into(), serde_json::json!(3));
        cp3.channel_versions
            .insert("b".into(), serde_json::json!(1));
        let mut new_versions = HashMap::new();
        new_versions.insert("a".into(), serde_json::json!(3));
        saver.put(&config, cp3, &metadata, &new_versions).unwrap();
        let tuple = saver.get_tuple(&config).unwrap().unwrap();
        assert_eq!(tuple.checkpoint.channel_values.get("a"), None);
        assert_eq!(
            tuple.checkpoint.channel_values.get("b"),
            Some(&serde_json::json!(1))
        );
    }
}
