use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::Row;

use langgraph_checkpoint::checkpoint::base::{
    get_checkpoint_id, get_checkpoint_metadata, writes_idx_map, BaseCheckpointSaver,
};
use langgraph_checkpoint::checkpoint::types::*;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_checkpoint::error::CheckpointError;
use langgraph_checkpoint::serde::base::SerializerProtocol;
use langgraph_checkpoint::serde::jsonplus::JsonPlusSerializer;

use crate::queries::*;

/// Async SQLite checkpoint saver using sqlx.
///
/// Uses a three-table schema (`checkpoints`, `checkpoint_blobs`,
/// `checkpoint_writes`) consistent with the Postgres implementation.
pub struct SqliteSaver {
    pool: SqlitePool,
    serde: Arc<dyn SerializerProtocol>,
}

impl SqliteSaver {
    /// Create a new SqliteSaver from an existing connection pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            serde: Arc::new(JsonPlusSerializer::new()),
        }
    }

    /// Create a new SqliteSaver with a custom serializer.
    pub fn with_serde(pool: SqlitePool, serde: Arc<dyn SerializerProtocol>) -> Self {
        Self { pool, serde }
    }

    /// Create a SqliteSaver from a connection string.
    ///
    /// Accepts standard sqlx URIs such as `"sqlite::memory:"` or
    /// `"sqlite:./checkpoints.db"`. The database file is created if
    /// it does not exist, and WAL journal mode is enabled.
    pub async fn from_conn_string(conn_string: &str) -> Result<Self, CheckpointError> {
        let opts = SqliteConnectOptions::from_str(conn_string)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(opts)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        Ok(Self::new(pool))
    }

    /// Run migrations to set up the checkpoint schema. Idempotent.
    pub async fn setup(&self) -> Result<(), CheckpointError> {
        // Bootstrap migrations table first
        sqlx::query(MIGRATIONS[0])
            .execute(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let row: Option<(i64,)> = sqlx::query_as(
            "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let version = row.map(|(v,)| v).unwrap_or(-1);

        for (i, migration) in MIGRATIONS.iter().enumerate() {
            let v = i as i64;
            if v > version {
                sqlx::query(migration)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                sqlx::query("INSERT INTO checkpoint_migrations (v) VALUES (?1)")
                    .bind(v)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get the underlying connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Helper: build a RunnableConfig referring to a specific checkpoint.
    fn make_config(thread_id: &str, checkpoint_ns: &str, checkpoint_id: &str) -> RunnableConfig {
        serde_json::from_value(serde_json::json!({
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }))
        .unwrap_or_default()
    }

    /// Convert a checkpoint row into a `CheckpointTuple`. Channel values
    /// from the row's JSON `checkpoint` column are kept as-is; the
    /// authoritative blob storage is reconciled separately by the caller.
    fn row_to_tuple(row: &SqliteRow) -> Result<CheckpointTuple, CheckpointError> {
        let thread_id: String = row.get("thread_id");
        let checkpoint_ns: String = row.get("checkpoint_ns");
        let checkpoint_text: String = row.get("checkpoint");
        let metadata_text: String = row.get("metadata");

        let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_text)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_text)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let parent_checkpoint_id: Option<String> = row.try_get("parent_checkpoint_id").ok();
        let parent_config = parent_checkpoint_id.map(|pid| {
            Self::make_config(&thread_id, &checkpoint_ns, &pid)
        });

        let tuple_config = Self::make_config(&thread_id, &checkpoint_ns, &checkpoint.id);

        Ok(CheckpointTuple {
            config: tuple_config,
            checkpoint,
            metadata,
            parent_config,
            pending_writes: None,
        })
    }

    /// Load blobs for a checkpoint and merge them into the channel_values map.
    async fn load_blobs(
        &self,
        thread_id: &str,
        checkpoint_ns: &str,
        checkpoint_id: &str,
    ) -> Result<HashMap<String, JsonValue>, CheckpointError> {
        let rows = sqlx::query(SELECT_BLOBS_SQL)
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let mut values: HashMap<String, JsonValue> = HashMap::new();
        for row in rows {
            let channel: String = row.get("channel");
            let type_tag: String = row.get("type");
            let blob: Option<Vec<u8>> = row.try_get("blob").ok();

            if type_tag == "empty" || blob.is_none() {
                continue;
            }
            let bytes = blob.unwrap();
            let val = match self.serde.loads_typed(&type_tag, &bytes) {
                Ok(any_val) => any_to_json(any_val),
                Err(_) => continue,
            };
            values.insert(channel, val);
        }
        Ok(values)
    }

    /// Load pending writes for a checkpoint.
    async fn load_writes(
        &self,
        thread_id: &str,
        checkpoint_ns: &str,
        checkpoint_id: &str,
    ) -> Result<Vec<PendingWrite>, CheckpointError> {
        let rows = sqlx::query(SELECT_WRITES_SQL)
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let mut writes = Vec::with_capacity(rows.len());
        for row in rows {
            let task_id: String = row.get("task_id");
            let channel: String = row.get("channel");
            let type_tag: Option<String> = row.try_get("type").ok();
            let blob: Option<Vec<u8>> = row.try_get("blob").ok();

            let value = match (type_tag.as_deref(), blob) {
                (Some(tag), Some(bytes)) => match self.serde.loads_typed(tag, &bytes) {
                    Ok(any_val) => any_to_json(any_val),
                    Err(_) => JsonValue::Null,
                },
                _ => JsonValue::Null,
            };
            writes.push((task_id, channel, value));
        }
        Ok(writes)
    }

    /// Serialize new channel values into blob rows.
    fn dump_blobs(
        &self,
        thread_id: &str,
        checkpoint_ns: &str,
        values: &HashMap<String, JsonValue>,
        versions: &ChannelVersions,
    ) -> Vec<(String, String, String, String, String, Option<Vec<u8>>)> {
        let mut result = Vec::new();
        for (channel, ver) in versions {
            let ver_str = match ver {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => n.to_string(),
                _ => continue,
            };
            if let Some(val) = values.get(channel) {
                if let Ok((type_tag, blob)) = self.serde.dumps_typed(val) {
                    result.push((
                        thread_id.to_string(),
                        checkpoint_ns.to_string(),
                        channel.clone(),
                        ver_str,
                        type_tag,
                        Some(blob),
                    ));
                }
            } else {
                result.push((
                    thread_id.to_string(),
                    checkpoint_ns.to_string(),
                    channel.clone(),
                    ver_str,
                    "empty".to_string(),
                    None,
                ));
            }
        }
        result
    }

    /// Async list method.
    pub async fn alist(
        &self,
        config: Option<&RunnableConfig>,
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        // Build dynamic WHERE with positional parameters (?1, ?2, ...).
        let mut conditions: Vec<String> = Vec::new();
        let mut binds: Vec<String> = Vec::new();

        if let Some(cfg) = config {
            if let Some(thread_id) = cfg
                .get("configurable")
                .and_then(|c| c.get("thread_id"))
                .and_then(|v| v.as_str())
            {
                conditions.push(format!("thread_id = ?{}", binds.len() + 1));
                binds.push(thread_id.to_string());
            }
            if let Some(ns) = cfg
                .get("configurable")
                .and_then(|c| c.get("checkpoint_ns"))
                .and_then(|v| v.as_str())
            {
                conditions.push(format!("checkpoint_ns = ?{}", binds.len() + 1));
                binds.push(ns.to_string());
            }
            if let Some(cid) = get_checkpoint_id(cfg) {
                conditions.push(format!("checkpoint_id = ?{}", binds.len() + 1));
                binds.push(cid);
            }
        }

        // Metadata filter: emits one `json_extract(metadata, '$.key') =
        // json_extract(?, '$')` clause per key. Both sides go through
        // `json_extract` so comparison is type-uniform regardless of
        // whether the value is a string, number, bool, array, or object.
        // Filter keys are validated against an allow-list to prevent
        // SQL injection via the inlined `'$.{key}'` JSON path.
        if let Some(meta_filter) = filter {
            for (key, value) in meta_filter {
                validate_filter_key(key)?;
                conditions.push(format!(
                    "json_extract(metadata, '$.{}') = json_extract(?{}, '$')",
                    key,
                    binds.len() + 1
                ));
                binds.push(serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()));
            }
        }

        if let Some(before_cfg) = before {
            if let Some(before_id) = get_checkpoint_id(before_cfg) {
                conditions.push(format!("checkpoint_id < ?{}", binds.len() + 1));
                binds.push(before_id);
            }
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let mut query = format!(
            "{} {} ORDER BY checkpoint_id DESC",
            SELECT_CHECKPOINT_SQL, where_clause
        );
        if let Some(lim) = limit {
            query.push_str(&format!(" LIMIT {}", lim));
        }

        let mut q = sqlx::query(&query);
        for b in &binds {
            q = q.bind(b.as_str());
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let mut tuple = Self::row_to_tuple(&row)?;
            // Reconcile channel values from blobs.
            let thread_id = row.get::<String, _>("thread_id");
            let ns = row.get::<String, _>("checkpoint_ns");
            let cid = tuple.checkpoint.id.clone();
            let blob_values = self.load_blobs(&thread_id, &ns, &cid).await?;
            if !blob_values.is_empty() {
                tuple.checkpoint.channel_values = blob_values;
            }
            tuple.pending_writes = Some(self.load_writes(&thread_id, &ns, &cid).await?);
            results.push(tuple);
        }
        Ok(results)
    }
}

/// Bridge an async future to a sync caller.
///
/// **Local triage** — see PR notes. The trait's sync methods (`get_tuple`,
/// `put`, `put_writes`, `delete_thread`, `list`) get invoked from inside
/// `langgraph::graph::state::run_pregel_inner`, which is itself an
/// `async fn`. Calling `Handle::block_on` from within a runtime panics
/// with *"Cannot start a runtime from within a runtime"*. The proper
/// fix is to make the graph runner call `aget_tuple`/`aput`/etc.; that
/// touches `langgraph` and is out of scope for this crate.
///
/// As a stopgap we use `block_in_place` to escape the worker thread and
/// then drive the future via the existing handle. **This requires a
/// multi-thread runtime** — calling sync saver methods from a
/// `current_thread` runtime will still panic. The `langgraph-checkpoint-postgres`
/// crate has the identical limitation today.
fn block_on_in_runtime<F, T>(future: F) -> Result<T, CheckpointError>
where
    F: std::future::Future<Output = Result<T, CheckpointError>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(future)),
        Err(_) => {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            rt.block_on(future)
        }
    }
}

/// Validate a metadata filter key. Allowed: ASCII letters, digits,
/// dot, underscore, hyphen. Empty strings rejected. The validated key
/// is interpolated into the SQL JSON path (`'$.{key}'`) so anything
/// that could break out of the literal must be rejected here.
fn validate_filter_key(key: &str) -> Result<(), CheckpointError> {
    if key.is_empty()
        || key
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-'))
    {
        return Err(CheckpointError::Config(format!(
            "invalid metadata filter key: {:?}",
            key
        )));
    }
    Ok(())
}

/// Best-effort conversion of a deserialized value (Box<dyn Any>) back into JSON.
fn any_to_json(val: Box<dyn std::any::Any>) -> JsonValue {
    if val.is::<JsonValue>() {
        *val.downcast::<JsonValue>().unwrap()
    } else if val.is::<String>() {
        JsonValue::String(*val.downcast::<String>().unwrap())
    } else if val.is::<Vec<u8>>() {
        let b = val.downcast::<Vec<u8>>().unwrap();
        JsonValue::Array(b.into_iter().map(|byte: u8| JsonValue::Number(byte.into())).collect())
    } else {
        JsonValue::Null
    }
}

#[async_trait]
impl BaseCheckpointSaver for SqliteSaver {
    fn get_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        block_on_in_runtime(self.aget_tuple(config))
    }

    fn list(
        &self,
        config: Option<&RunnableConfig>,
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        block_on_in_runtime(self.alist(config, filter, before, limit))
    }

    fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: &Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        block_on_in_runtime(self.aput(config, checkpoint, metadata, new_versions))
    }

    fn put_writes(
        &self,
        config: &RunnableConfig,
        writes: &[(String, String, JsonValue)],
        task_id: &str,
        task_path: &str,
    ) -> Result<(), CheckpointError> {
        block_on_in_runtime(self.aput_writes(
            config,
            writes.to_vec(),
            task_id.to_string(),
            task_path.to_string(),
        ))
    }

    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        block_on_in_runtime(self.adelete_thread(thread_id.to_string()))
    }

    async fn aget_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        let thread_id = config
            .get("configurable")
            .and_then(|c| c.get("thread_id"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckpointError::Config("missing thread_id".into()))?;

        let checkpoint_ns = config
            .get("configurable")
            .and_then(|c| c.get("checkpoint_ns"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let checkpoint_id = get_checkpoint_id(config);

        let row = if let Some(cid) = &checkpoint_id {
            sqlx::query(&format!(
                "{} WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3",
                SELECT_CHECKPOINT_SQL
            ))
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(cid.as_str())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?
        } else {
            sqlx::query(&format!(
                "{} WHERE thread_id = ?1 AND checkpoint_ns = ?2 ORDER BY checkpoint_id DESC LIMIT 1",
                SELECT_CHECKPOINT_SQL
            ))
            .bind(thread_id)
            .bind(checkpoint_ns)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?
        };

        let row = match row {
            Some(r) => r,
            None => return Ok(None),
        };

        let mut tuple = Self::row_to_tuple(&row)?;
        let cid = tuple.checkpoint.id.clone();
        let blob_values = self.load_blobs(thread_id, checkpoint_ns, &cid).await?;
        if !blob_values.is_empty() {
            tuple.checkpoint.channel_values = blob_values;
        }
        tuple.pending_writes = Some(self.load_writes(thread_id, checkpoint_ns, &cid).await?);
        Ok(Some(tuple))
    }

    async fn aput(
        &self,
        config: &RunnableConfig,
        checkpoint: &Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        let configurable = config.get("configurable").cloned().unwrap_or_default();
        let thread_id = configurable
            .get("thread_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckpointError::Config("missing thread_id".into()))?;
        let checkpoint_ns = configurable
            .get("checkpoint_ns")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let parent_checkpoint_id: Option<String> = configurable
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let next_config = Self::make_config(thread_id, checkpoint_ns, &checkpoint.id);

        // Strip channel_values from the JSON checkpoint payload to avoid
        // duplicating them in the row body — they live in checkpoint_blobs.
        let mut checkpoint_value = serde_json::to_value(checkpoint)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        if let Some(obj) = checkpoint_value.as_object_mut() {
            obj.insert("channel_values".to_string(), JsonValue::Object(Default::default()));
        }
        let checkpoint_text = serde_json::to_string(&checkpoint_value)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        // Merge config-level fields (e.g. `langgraph_step`) into the
        // metadata before persisting, so `list(filter=...)` over those
        // fields can find the row. Mirrors Python's get_checkpoint_metadata.
        let merged_metadata = get_checkpoint_metadata(config, metadata);
        let metadata_text = serde_json::to_string(&merged_metadata)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        // Upsert blobs
        let blobs = self.dump_blobs(
            thread_id,
            checkpoint_ns,
            &checkpoint.channel_values,
            new_versions,
        );
        for (tid, cns, channel, version, type_tag, blob) in &blobs {
            sqlx::query(UPSERT_CHECKPOINT_BLOBS_SQL)
                .bind(tid.as_str())
                .bind(cns.as_str())
                .bind(channel.as_str())
                .bind(version.as_str())
                .bind(type_tag.as_str())
                .bind(blob.as_deref())
                .execute(&mut *tx)
                .await
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        }

        // Upsert checkpoint row
        sqlx::query(UPSERT_CHECKPOINTS_SQL)
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint.id.as_str())
            .bind(parent_checkpoint_id.as_deref())
            .bind(checkpoint_text.as_str())
            .bind(metadata_text.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        Ok(next_config)
    }

    async fn aput_writes(
        &self,
        config: &RunnableConfig,
        writes: Vec<(String, String, JsonValue)>,
        task_id: String,
        task_path: String,
    ) -> Result<(), CheckpointError> {
        let configurable = config.get("configurable").cloned().unwrap_or_default();
        let thread_id = configurable
            .get("thread_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckpointError::Config("missing thread_id".into()))?;
        let checkpoint_ns = configurable
            .get("checkpoint_ns")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // NOTE: align with langgraph-checkpoint-postgres which silently
        // defaults to empty string when checkpoint_id is missing. The
        // graph runner currently calls put_writes with the *input*
        // config (not the new config returned by put), so on the first
        // step checkpoint_id is absent. Erroring here would crash the
        // run; defaulting matches existing behavior. This does NOT
        // semantically fix interrupt/resume — pending writes still get
        // attached to checkpoint_id="" and won't be reachable from
        // get_tuple of the real latest checkpoint. Tracked as a
        // cross-crate issue (graph runner should pass the post-put
        // config, or saver should resolve latest checkpoint here).
        let checkpoint_id = configurable
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let idx_map = writes_idx_map();
        let use_upsert = writes
            .iter()
            .all(|(channel, _, _)| idx_map.contains_key(channel.as_str()));

        let query = if use_upsert {
            UPSERT_CHECKPOINT_WRITES_SQL
        } else {
            INSERT_CHECKPOINT_WRITES_SQL
        };

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        for (idx, (_task_id_in_tuple, channel, value)) in writes.iter().enumerate() {
            let idx_val: i64 = idx_map
                .get(channel.as_str())
                .copied()
                .unwrap_or(idx as i64);

            let (type_tag, blob) = match self.serde.dumps_typed(value) {
                Ok(pair) => pair,
                Err(_) => continue,
            };

            sqlx::query(query)
                .bind(thread_id)
                .bind(checkpoint_ns)
                .bind(checkpoint_id)
                .bind(task_id.as_str())
                .bind(task_path.as_str())
                .bind(idx_val)
                .bind(channel.as_str())
                .bind(type_tag.as_str())
                .bind(blob.as_slice())
                .execute(&mut *tx)
                .await
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn adelete_thread(&self, thread_id: String) -> Result<(), CheckpointError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoints WHERE thread_id = ?1")
            .bind(thread_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoint_blobs WHERE thread_id = ?1")
            .bind(thread_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoint_writes WHERE thread_id = ?1")
            .bind(thread_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        tx.commit()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    async fn fresh_saver() -> SqliteSaver {
        let saver = SqliteSaver::from_conn_string("sqlite::memory:")
            .await
            .expect("connect to in-memory sqlite");
        saver.setup().await.expect("setup migrations");
        saver
    }

    fn config_for(thread_id: &str) -> RunnableConfig {
        serde_json::from_value(serde_json::json!({
            "configurable": { "thread_id": thread_id, "checkpoint_ns": "" }
        }))
        .unwrap()
    }

    fn config_with_id(thread_id: &str, checkpoint_id: &str) -> RunnableConfig {
        serde_json::from_value(serde_json::json!({
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }))
        .unwrap()
    }

    fn make_checkpoint(channel_values: Vec<(&str, JsonValue)>) -> (Checkpoint, ChannelVersions) {
        let mut cp = Checkpoint::empty();
        let mut versions: ChannelVersions = HashMap::new();
        for (k, v) in channel_values {
            cp.channel_values.insert(k.to_string(), v);
            cp.channel_versions
                .insert(k.to_string(), JsonValue::Number(1.into()));
            versions.insert(k.to_string(), JsonValue::Number(1.into()));
        }
        (cp, versions)
    }

    #[tokio::test]
    async fn test_setup_is_idempotent() {
        let saver = fresh_saver().await;
        // calling setup again should not error
        saver.setup().await.expect("second setup");
    }

    #[tokio::test]
    async fn test_get_tuple_returns_none_when_empty() {
        let saver = fresh_saver().await;
        let cfg = config_for("missing");
        let result = saver.aget_tuple(&cfg).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_put_then_get_roundtrip() {
        let saver = fresh_saver().await;
        let (cp, versions) = make_checkpoint(vec![
            ("messages", serde_json::json!(["hello", "world"])),
            ("counter", serde_json::json!(7)),
        ]);
        let cfg = config_for("thread-A");
        let metadata = CheckpointMetadata {
            source: Some(CheckpointSource::Loop),
            step: Some(3),
            ..Default::default()
        };

        let next = saver.aput(&cfg, &cp, &metadata, &versions).await.unwrap();

        // The returned config should reference the new checkpoint id
        let returned_cid = next
            .get("configurable")
            .and_then(|c| c.get("checkpoint_id"))
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(returned_cid, cp.id);

        // Fetch back and compare
        let tuple = saver.aget_tuple(&cfg).await.unwrap().expect("tuple exists");
        assert_eq!(tuple.checkpoint.id, cp.id);
        assert_eq!(tuple.metadata.step, Some(3));
        assert_eq!(
            tuple.checkpoint.channel_values.get("messages"),
            Some(&serde_json::json!(["hello", "world"]))
        );
        assert_eq!(
            tuple.checkpoint.channel_values.get("counter"),
            Some(&serde_json::json!(7))
        );
    }

    #[tokio::test]
    async fn test_put_writes_and_pending_writes_round_trip() {
        let saver = fresh_saver().await;
        let (cp, versions) = make_checkpoint(vec![("a", serde_json::json!(1))]);
        let cfg = config_for("thread-W");
        saver
            .aput(&cfg, &cp, &CheckpointMetadata::default(), &versions)
            .await
            .unwrap();

        let cfg_with_id = config_with_id("thread-W", &cp.id);
        let writes = vec![
            ("ch1".to_string(), "task-1".to_string(), serde_json::json!("v1")),
            ("ch2".to_string(), "task-1".to_string(), serde_json::json!(42)),
        ];
        saver
            .aput_writes(&cfg_with_id, writes, "task-1".into(), "".into())
            .await
            .unwrap();

        let tuple = saver.aget_tuple(&cfg_with_id).await.unwrap().unwrap();
        let pending = tuple.pending_writes.expect("pending writes loaded");
        assert_eq!(pending.len(), 2);
        // Order: by task_path, task_id, idx
        assert_eq!(pending[0].1, "ch1");
        assert_eq!(pending[1].1, "ch2");
        assert_eq!(pending[1].2, serde_json::json!(42));
    }

    #[tokio::test]
    async fn test_list_orders_descending_and_respects_limit() {
        let saver = fresh_saver().await;
        let cfg = config_for("thread-L");
        let mut ids = Vec::new();
        for i in 0..3 {
            let (cp, versions) = make_checkpoint(vec![("x", serde_json::json!(i))]);
            ids.push(cp.id.clone());
            saver
                .aput(&cfg, &cp, &CheckpointMetadata::default(), &versions)
                .await
                .unwrap();
        }

        let all = saver.alist(Some(&cfg), None, None, None).await.unwrap();
        assert_eq!(all.len(), 3);
        // ORDER BY checkpoint_id DESC — verify returned ids are in
        // descending lexicographic order (not tied to insertion order,
        // since UUIDv7 within the same millisecond is not monotonic).
        for w in all.windows(2) {
            assert!(w[0].checkpoint.id >= w[1].checkpoint.id);
        }
        // All three checkpoint ids should appear in the result set
        let returned_ids: std::collections::HashSet<_> =
            all.iter().map(|t| t.checkpoint.id.clone()).collect();
        for id in &ids {
            assert!(returned_ids.contains(id));
        }

        let limited = saver.alist(Some(&cfg), None, None, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_thread_removes_all_data() {
        let saver = fresh_saver().await;
        let (cp, versions) = make_checkpoint(vec![("x", serde_json::json!(1))]);
        let cfg = config_for("thread-D");
        saver
            .aput(&cfg, &cp, &CheckpointMetadata::default(), &versions)
            .await
            .unwrap();
        let cfg_with_id = config_with_id("thread-D", &cp.id);
        saver
            .aput_writes(
                &cfg_with_id,
                vec![("ch".into(), "task".into(), serde_json::json!("v"))],
                "task".into(),
                "".into(),
            )
            .await
            .unwrap();

        saver.adelete_thread("thread-D".into()).await.unwrap();
        assert!(saver.aget_tuple(&cfg).await.unwrap().is_none());
        let listed = saver.alist(Some(&cfg), None, None, None).await.unwrap();
        assert!(listed.is_empty());
    }

    #[tokio::test]
    async fn test_value_updates_when_version_increments() {
        // Blob storage is keyed by (channel, version). Ensure that when
        // a caller bumps the version, the new value is what's read back.
        let saver = fresh_saver().await;
        let cfg = config_for("thread-V");

        // First put: counter=1 at version 1
        let mut cp1 = Checkpoint::empty();
        cp1.channel_values
            .insert("counter".into(), JsonValue::Number(1.into()));
        cp1.channel_versions
            .insert("counter".into(), JsonValue::Number(1.into()));
        let mut versions1: ChannelVersions = HashMap::new();
        versions1.insert("counter".into(), JsonValue::Number(1.into()));
        saver
            .aput(&cfg, &cp1, &CheckpointMetadata::default(), &versions1)
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(2)).await;

        // Second put: counter=99 at version 2 — fresh blob row.
        let mut cp2 = Checkpoint::empty();
        cp2.channel_values
            .insert("counter".into(), JsonValue::Number(99.into()));
        cp2.channel_versions
            .insert("counter".into(), JsonValue::Number(2.into()));
        let mut versions2: ChannelVersions = HashMap::new();
        versions2.insert("counter".into(), JsonValue::Number(2.into()));
        saver
            .aput(&cfg, &cp2, &CheckpointMetadata::default(), &versions2)
            .await
            .unwrap();

        let cfg_cp2 = config_with_id("thread-V", &cp2.id);
        let tuple = saver.aget_tuple(&cfg_cp2).await.unwrap().unwrap();
        assert_eq!(
            tuple.checkpoint.channel_values.get("counter"),
            Some(&JsonValue::Number(99.into()))
        );

        // The earlier checkpoint should still see its own value.
        let cfg_cp1 = config_with_id("thread-V", &cp1.id);
        let earlier = saver.aget_tuple(&cfg_cp1).await.unwrap().unwrap();
        assert_eq!(
            earlier.checkpoint.channel_values.get("counter"),
            Some(&JsonValue::Number(1.into()))
        );
    }

    #[tokio::test]
    async fn test_metadata_filter_returns_only_matching_rows() {
        let saver = fresh_saver().await;
        let cfg = config_for("thread-F");

        // Three checkpoints with distinct `source` and `step` metadata.
        for (source, step, val) in [
            (CheckpointSource::Input, 0, "a"),
            (CheckpointSource::Loop, 1, "b"),
            (CheckpointSource::Loop, 2, "c"),
        ] {
            let (cp, vers) = make_checkpoint(vec![("x", serde_json::json!(val))]);
            let meta = CheckpointMetadata {
                source: Some(source),
                step: Some(step),
                ..Default::default()
            };
            saver.aput(&cfg, &cp, &meta, &vers).await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }

        // Filter source = "loop" → 2 results
        let mut filter = HashMap::new();
        filter.insert("source".into(), serde_json::json!("loop"));
        let loop_only = saver
            .alist(Some(&cfg), Some(&filter), None, None)
            .await
            .unwrap();
        assert_eq!(loop_only.len(), 2);
        for t in &loop_only {
            assert_eq!(t.metadata.source, Some(CheckpointSource::Loop));
        }

        // Filter step = 1 → 1 result
        let mut filter = HashMap::new();
        filter.insert("step".into(), serde_json::json!(1));
        let step_one = saver
            .alist(Some(&cfg), Some(&filter), None, None)
            .await
            .unwrap();
        assert_eq!(step_one.len(), 1);
        assert_eq!(step_one[0].metadata.step, Some(1));

        // Combined filter: source = "loop" AND step = 2 → 1 result
        let mut filter = HashMap::new();
        filter.insert("source".into(), serde_json::json!("loop"));
        filter.insert("step".into(), serde_json::json!(2));
        let combined = saver
            .alist(Some(&cfg), Some(&filter), None, None)
            .await
            .unwrap();
        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].metadata.step, Some(2));
    }

    #[test]
    fn test_validate_filter_key_rejects_injection_attempts() {
        // Valid keys
        assert!(validate_filter_key("source").is_ok());
        assert!(validate_filter_key("nested.field").is_ok());
        assert!(validate_filter_key("snake_case").is_ok());
        assert!(validate_filter_key("kebab-case").is_ok());
        assert!(validate_filter_key("Mixed123").is_ok());

        // Invalid: empty, quotes, semicolons, brackets, spaces, unicode
        assert!(validate_filter_key("").is_err());
        assert!(validate_filter_key("source'; DROP TABLE--").is_err());
        assert!(validate_filter_key("a\"b").is_err());
        assert!(validate_filter_key("a b").is_err());
        assert!(validate_filter_key("[admin]").is_err());
        assert!(validate_filter_key("中文").is_err());
    }

    #[tokio::test]
    async fn test_config_langgraph_step_merged_into_metadata() {
        // When the caller includes `langgraph_step` in configurable, the
        // saver should fold it into the persisted metadata (so list-with-filter
        // can later find rows by step).
        let saver = fresh_saver().await;
        let cfg: RunnableConfig = serde_json::from_value(serde_json::json!({
            "configurable": {
                "thread_id": "thread-M",
                "checkpoint_ns": "",
                "langgraph_step": 7
            }
        }))
        .unwrap();

        let (cp, vers) = make_checkpoint(vec![("x", serde_json::json!(1))]);
        // Metadata passed in does NOT have step set — it should be filled
        // from the config.
        saver
            .aput(&cfg, &cp, &CheckpointMetadata::default(), &vers)
            .await
            .unwrap();

        let cfg_with_id = config_with_id("thread-M", &cp.id);
        let tuple = saver.aget_tuple(&cfg_with_id).await.unwrap().unwrap();
        assert_eq!(tuple.metadata.step, Some(7));
    }

    /// Verifies the sync wrapper does not panic when called from inside
    /// a multi-thread tokio runtime (the situation that occurs when the
    /// graph runner — itself an async fn — invokes `cp.get_tuple(...)`).
    /// See `block_on_in_runtime` doc comment.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_sync_methods_work_inside_multi_thread_runtime() {
        let saver = fresh_saver().await;
        let saver = std::sync::Arc::new(saver);
        let cfg = config_for("thread-S");

        // Drive sync `put` and `get_tuple` on a blocking task — this is
        // what `block_in_place` is designed to guard against.
        let (cp, vers) = make_checkpoint(vec![("k", serde_json::json!("v"))]);
        let s2 = saver.clone();
        let cfg2 = cfg.clone();
        let cp_clone = cp.clone();
        let vers_clone = vers.clone();
        let put_result = tokio::task::spawn_blocking(move || {
            s2.put(&cfg2, &cp_clone, &CheckpointMetadata::default(), &vers_clone)
        })
        .await
        .unwrap();
        assert!(put_result.is_ok());

        let s3 = saver.clone();
        let cfg3 = cfg.clone();
        let get_result = tokio::task::spawn_blocking(move || s3.get_tuple(&cfg3))
            .await
            .unwrap()
            .unwrap();
        assert!(get_result.is_some());
        assert_eq!(get_result.unwrap().checkpoint.id, cp.id);
    }

    #[tokio::test]
    async fn test_parent_config_links_checkpoints() {
        let saver = fresh_saver().await;
        let (cp1, vers1) = make_checkpoint(vec![("x", serde_json::json!("a"))]);
        let cfg = config_for("thread-P");
        let next1 = saver
            .aput(&cfg, &cp1, &CheckpointMetadata::default(), &vers1)
            .await
            .unwrap();

        // Sleep briefly so UUIDv7 timestamps differ; otherwise two
        // checkpoints created within the same millisecond can sort
        // unpredictably.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;

        // Second put using next1 as the parent config — its checkpoint_id
        // becomes the parent_checkpoint_id of cp2.
        let (cp2, vers2) = make_checkpoint(vec![("x", serde_json::json!("b"))]);
        saver
            .aput(&next1, &cp2, &CheckpointMetadata::default(), &vers2)
            .await
            .unwrap();

        // Look up cp2 explicitly to avoid relying on lex ordering.
        let cfg_cp2 = config_with_id("thread-P", &cp2.id);
        let latest = saver.aget_tuple(&cfg_cp2).await.unwrap().unwrap();
        assert_eq!(latest.checkpoint.id, cp2.id);
        let parent = latest.parent_config.expect("parent_config present");
        let parent_id = parent
            .get("configurable")
            .and_then(|c| c.get("checkpoint_id"))
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(parent_id, cp1.id);
    }
}
