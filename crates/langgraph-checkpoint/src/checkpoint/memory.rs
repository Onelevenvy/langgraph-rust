use super::base::BaseCheckpointSaver;
use super::types::*;
use crate::config::RunnableConfig;
use crate::error::CheckpointError;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

type StorageKey = (String, String, String); // (thread_id, checkpoint_ns, checkpoint_id)
type WriteKey = (String, String, String, i64); // (thread_id, checkpoint_ns, checkpoint_id, idx)

/// In-memory checkpoint saver for testing and development.
///
/// Stores checkpoints, blobs, and writes in memory using DashMap
/// for concurrent access.
pub struct InMemorySaver {
    // (thread_id, checkpoint_ns, checkpoint_id) -> (checkpoint_json, metadata_json, parent_checkpoint_id)
    storage: RwLock<HashMap<StorageKey, (JsonValue, JsonValue, Option<String>)>>,
    // (thread_id, checkpoint_ns, checkpoint_id, idx) -> (task_id, channel, value_json, task_path)
    writes: RwLock<HashMap<WriteKey, (String, String, JsonValue, String)>>,
}

impl InMemorySaver {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(HashMap::new()),
            writes: RwLock::new(HashMap::new()),
        }
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
        let storage = self.storage.read();

        // Find the checkpoint
        let key = if let Some(ref cid) = checkpoint_id {
            (thread_id.clone(), checkpoint_ns.clone(), cid.clone())
        } else {
            // Find the latest checkpoint for this thread/ns
            let candidates: Vec<_> = storage
                .keys()
                .filter(|(tid, ns, _)| tid == &thread_id && ns == &checkpoint_ns)
                .collect();

            match candidates.into_iter().max_by_key(|(_, _, cid)| cid.clone()) {
                Some((tid, ns, cid)) => (tid.clone(), ns.clone(), cid.clone()),
                None => return Ok(None),
            }
        };

        let (checkpoint_json, metadata_json, parent_cid) = match storage.get(&key) {
            Some(v) => v.clone(),
            None => return Ok(None),
        };

        let checkpoint: Checkpoint = serde_json::from_value(checkpoint_json)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata: CheckpointMetadata = serde_json::from_value(metadata_json)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

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
            .filter(|((tid, ns, cid, _), _)| tid == &key.0 && ns == &key.1 && cid == &key.2)
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
                        "checkpoint_id": key.2,
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
        let storage = self.storage.read();

        let (thread_id, checkpoint_ns) = match config {
            Some(c) => {
                let (tid, ns, _) = Self::config_to_ids(c);
                (tid, ns)
            }
            None => (String::new(), String::new()),
        };

        let before_id = before.and_then(|c| Self::config_to_ids(c).2);

        let mut entries: Vec<_> = storage
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
        for ((tid, ns, cid), (checkpoint_json, metadata_json, parent_cid)) in entries {
            // Apply filter
            if let Some(filter) = filter {
                let metadata_val: JsonValue = metadata_json.clone();
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

        // Get the current parent
        let parent_id = {
            let storage = self.storage.read();
            storage
                .keys()
                .filter(|(tid, ns, _)| tid == &thread_id && ns == &checkpoint_ns)
                .max_by_key(|(_, _, cid)| cid.clone())
                .map(|(_, _, cid)| cid.clone())
        };

        let key = (
            thread_id.clone(),
            checkpoint_ns.clone(),
            checkpoint.id.clone(),
        );
        self.storage
            .write()
            .insert(key, (checkpoint_json, metadata_json, parent_id));

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
            // write_tuple is (String, String, JsonValue) - (channel, type_or_extra, value)
            writes_map.insert(
                key,
                (
                    task_id.to_string(),
                    write_tuple.0.clone(),
                    write_tuple.2.clone(),
                    task_path.to_string(),
                ),
            );
        }
        Ok(())
    }

    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        self.storage
            .write()
            .retain(|(tid, _, _), _| tid != thread_id);
        self.writes
            .write()
            .retain(|(tid, _, _, _), _| tid != thread_id);
        Ok(())
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
            .put(&config, &checkpoint, &metadata, &HashMap::new())
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
                .put(&config, &checkpoint, &metadata, &HashMap::new())
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
            .put(&config, &checkpoint, &metadata, &HashMap::new())
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
            .put(&config, &checkpoint, &metadata, &HashMap::new())
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
}
