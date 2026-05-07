use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use sqlx::postgres::{PgPool, PgPoolOptions, PgRow};
use sqlx::Row;

use langgraph_checkpoint::checkpoint::base::{get_checkpoint_id, writes_idx_map, BaseCheckpointSaver};
use langgraph_checkpoint::checkpoint::types::*;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_checkpoint::error::CheckpointError;
use langgraph_checkpoint::serde::base::SerializerProtocol;
use langgraph_checkpoint::serde::jsonplus::JsonPlusSerializer;

use crate::queries::*;

/// Blob row: (thread_id, checkpoint_ns, channel, version, type_tag, blob)
type BlobRow = (String, String, String, String, String, Option<Vec<u8>>);

/// Write row: (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type_tag, blob)
type WriteRow = (String, String, String, String, String, i32, String, String, Vec<u8>);

/// Helper: create a RunnableConfig from a JSON value.
fn config_from_json(val: serde_json::Value) -> RunnableConfig {
    serde_json::from_value(val).unwrap_or_default()
}

/// Helper: downcast Box<dyn Any> from loads_typed to JsonValue.
#[allow(dead_code)]
fn any_to_json(val: Box<dyn std::any::Any + Send + Sync>) -> JsonValue {
    if val.is::<JsonValue>() {
        *val.downcast::<JsonValue>().unwrap()
    } else if val.is::<String>() {
        JsonValue::String(*val.downcast::<String>().unwrap())
    } else if val.is::<Vec<u8>>() {
        let b = val.downcast::<Vec<u8>>().unwrap();
        JsonValue::Array(b.into_iter().map(|byte: u8| JsonValue::Number(byte.into())).collect())
    } else {
        // () and unknown types both map to Null
        JsonValue::Null
    }
}

/// Async Postgres checkpoint saver using sqlx.
pub struct PostgresSaver {
    pool: PgPool,
    serde: Arc<dyn SerializerProtocol>,
}

impl PostgresSaver {
    /// Create a new PostgresSaver from a connection pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            serde: Arc::new(JsonPlusSerializer::new()),
        }
    }

    /// Create a new PostgresSaver with a custom serializer.
    pub fn with_serde(pool: PgPool, serde: Arc<dyn SerializerProtocol>) -> Self {
        Self { pool, serde }
    }

    /// Create a PostgresSaver from a connection string.
    pub async fn from_conn_string(conn_string: &str) -> Result<Self, CheckpointError> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(conn_string)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        Ok(Self::new(pool))
    }

    /// Run migrations to set up the checkpoint schema.
    pub async fn setup(&self) -> Result<(), CheckpointError> {
        sqlx::query(MIGRATIONS[0])
            .execute(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let row: Option<(i32,)> = sqlx::query_as(
            "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let version = row.map(|(v,)| v).unwrap_or(-1);

        for (i, migration) in MIGRATIONS.iter().enumerate() {
            let v = i as i32;
            if v > version {
                sqlx::query(migration)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                sqlx::query("INSERT INTO checkpoint_migrations (v) VALUES ($1)")
                    .bind(v)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get the underlying connection pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Build a WHERE clause from config, filter, and before parameters.
    fn build_where_clause(
        config: Option<&RunnableConfig>,
        _filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
    ) -> (String, Vec<String>) {
        let mut wheres = Vec::new();
        let mut params = Vec::new();

        if let Some(config) = config {
            if let Some(thread_id) = config
                .get("configurable")
                .and_then(|c| c.get("thread_id"))
                .and_then(|v| v.as_str())
            {
                let idx = params.len() + 1;
                wheres.push(format!("thread_id = ${}", idx));
                params.push(thread_id.to_string());
            }

            if let Some(checkpoint_ns) = config
                .get("configurable")
                .and_then(|c| c.get("checkpoint_ns"))
                .and_then(|v| v.as_str())
            {
                let idx = params.len() + 1;
                wheres.push(format!("checkpoint_ns = ${}", idx));
                params.push(checkpoint_ns.to_string());
            }

            if let Some(checkpoint_id) = get_checkpoint_id(config) {
                let idx = params.len() + 1;
                wheres.push(format!("checkpoint_id = ${}", idx));
                params.push(checkpoint_id);
            }
        }

        if let Some(before) = before {
            if let Some(before_id) = get_checkpoint_id(before) {
                let idx = params.len() + 1;
                wheres.push(format!("checkpoint_id < ${}", idx));
                params.push(before_id);
            }
        }

        let where_clause = if wheres.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", wheres.join(" AND "))
        };

        (where_clause, params)
    }

    /// Serialize blobs for storage.
    fn dump_blobs(
        &self,
        thread_id: &str,
        checkpoint_ns: &str,
        values: &HashMap<String, JsonValue>,
        versions: &ChannelVersions,
    ) -> Vec<BlobRow> {
        let mut result = Vec::new();
        for (k, ver) in versions {
            let ver_str = match ver {
                JsonValue::String(s) => s.clone(),
                JsonValue::Number(n) => n.to_string(),
                _ => continue,
            };
            if let Some(val) = values.get(k) {
                if let Ok((type_tag, blob)) = self.serde.dumps_typed(val) {
                    result.push((
                        thread_id.to_string(),
                        checkpoint_ns.to_string(),
                        k.clone(),
                        ver_str,
                        type_tag,
                        Some(blob),
                    ));
                }
            } else {
                result.push((
                    thread_id.to_string(),
                    checkpoint_ns.to_string(),
                    k.clone(),
                    ver_str,
                    "empty".to_string(),
                    None,
                ));
            }
        }
        result
    }

    /// Serialize writes for storage.
    fn dump_writes(
        &self,
        thread_id: &str,
        checkpoint_ns: &str,
        checkpoint_id: &str,
        task_id: &str,
        task_path: &str,
        writes: &[(String, String, JsonValue)],
    ) -> Vec<WriteRow> {
        let idx_map = writes_idx_map();
        writes
            .iter()
            .enumerate()
            .filter_map(|(idx, (channel, _task_id, value))| {
                let idx_val = idx_map
                    .get(channel.as_str())
                    .copied()
                    .unwrap_or(idx as i64) as i32;
                if let Ok((type_tag, blob)) = self.serde.dumps_typed(value) {
                    Some((
                        thread_id.to_string(),
                        checkpoint_ns.to_string(),
                        checkpoint_id.to_string(),
                        task_id.to_string(),
                        task_path.to_string(),
                        idx_val,
                        channel.clone(),
                        type_tag,
                        blob,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Parse a checkpoint from a row and build a CheckpointTuple.
    fn row_to_tuple(row: &PgRow) -> Result<CheckpointTuple, CheckpointError> {
        let checkpoint_json: JsonValue = row.get("checkpoint");
        let metadata_json: JsonValue = row.get("metadata");

        let checkpoint: Checkpoint = serde_json::from_value(checkpoint_json)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata: CheckpointMetadata = serde_json::from_value(metadata_json)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let thread_id: String = row.get("thread_id");
        let checkpoint_ns: String = row.get("checkpoint_ns");

        let tuple_config = config_from_json(serde_json::json!({
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint.id,
            }
        }));

        let parent_config: Option<RunnableConfig> = row
            .get::<Option<String>, _>("parent_checkpoint_id")
            .map(|pid| {
                config_from_json(serde_json::json!({
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": pid,
                    }
                }))
            });

        Ok(CheckpointTuple {
            config: tuple_config,
            checkpoint,
            metadata,
            parent_config,
            pending_writes: None,
        })
    }
}

#[async_trait]
impl BaseCheckpointSaver for PostgresSaver {
    fn get_tuple(
        &self,
        config: &RunnableConfig,
    ) -> Result<Option<CheckpointTuple>, CheckpointError> {
        // For sync calls, try to use existing runtime or create one
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.aget_tuple(config)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                rt.block_on(self.aget_tuple(config))
            }
        }
    }

    fn list(
        &self,
        config: Option<&RunnableConfig>,
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.alist(config, filter, before, limit)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                rt.block_on(self.alist(config, filter, before, limit))
            }
        }
    }

    fn put(
        &self,
        config: &RunnableConfig,
        checkpoint: &Checkpoint,
        metadata: &CheckpointMetadata,
        new_versions: &ChannelVersions,
    ) -> Result<RunnableConfig, CheckpointError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.aput(config, checkpoint, metadata, new_versions)),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                rt.block_on(self.aput(config, checkpoint, metadata, new_versions))
            }
        }
    }

    fn put_writes(
        &self,
        config: &RunnableConfig,
        writes: &[(String, String, JsonValue)],
        task_id: &str,
        task_path: &str,
    ) -> Result<(), CheckpointError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.aput_writes(
                config,
                writes.to_vec(),
                task_id.to_string(),
                task_path.to_string(),
            )),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                rt.block_on(self.aput_writes(
                    config,
                    writes.to_vec(),
                    task_id.to_string(),
                    task_path.to_string(),
                ))
            }
        }
    }

    fn delete_thread(&self, thread_id: &str) -> Result<(), CheckpointError> {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(self.adelete_thread(thread_id.to_string())),
            Err(_) => {
                let rt = tokio::runtime::Runtime::new()
                    .map_err(|e| CheckpointError::Storage(e.to_string()))?;
                rt.block_on(self.adelete_thread(thread_id.to_string()))
            }
        }
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
                "{} WHERE thread_id = $1 AND checkpoint_ns = $2 AND checkpoint_id = $3",
                SELECT_SQL
            ))
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(cid.as_str())
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?
        } else {
            sqlx::query(&format!(
                "{} WHERE thread_id = $1 AND checkpoint_ns = $2 ORDER BY checkpoint_id DESC LIMIT 1",
                SELECT_SQL
            ))
            .bind(thread_id)
            .bind(checkpoint_ns)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?
        };

        match row {
            Some(row) => Ok(Some(Self::row_to_tuple(&row)?)),
            None => Ok(None),
        }
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

        let next_config = config_from_json(serde_json::json!({
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint.id,
            }
        }));

        let checkpoint_json = serde_json::to_value(checkpoint)
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        let metadata_json = serde_json::to_value(metadata)
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
                .execute(&self.pool)
                .await
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        }

        // Upsert checkpoint
        sqlx::query(UPSERT_CHECKPOINTS_SQL)
            .bind(thread_id)
            .bind(checkpoint_ns)
            .bind(checkpoint.id.as_str())
            .bind(parent_checkpoint_id.as_deref())
            .bind(&checkpoint_json)
            .bind(&metadata_json)
            .execute(&self.pool)
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

        let dump = self.dump_writes(
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            &task_id,
            &task_path,
            &writes,
        );

        for (tid, cns, cid, tid2, tpath, idx, channel, type_tag, blob) in &dump {
            sqlx::query(query)
                .bind(tid.as_str())
                .bind(cns.as_str())
                .bind(cid.as_str())
                .bind(tid2.as_str())
                .bind(tpath.as_str())
                .bind(*idx)
                .bind(channel.as_str())
                .bind(type_tag.as_str())
                .bind(blob.as_slice())
                .execute(&self.pool)
                .await
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    async fn adelete_thread(&self, thread_id: String) -> Result<(), CheckpointError> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoints WHERE thread_id = $1")
            .bind(thread_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoint_blobs WHERE thread_id = $1")
            .bind(thread_id.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        sqlx::query("DELETE FROM checkpoint_writes WHERE thread_id = $1")
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

/// Async list method for PostgresSaver.
impl PostgresSaver {
    pub async fn alist(
        &self,
        config: Option<&RunnableConfig>,
        filter: Option<&HashMap<String, JsonValue>>,
        before: Option<&RunnableConfig>,
        limit: Option<usize>,
    ) -> Result<Vec<CheckpointTuple>, CheckpointError> {
        let (where_clause, _params) = Self::build_where_clause(config, filter, before);
        let mut query = format!(
            "{} {} ORDER BY checkpoint_id DESC",
            SELECT_SQL, where_clause
        );

        if let Some(limit) = limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        // Build the query with bound params
        let mut q = sqlx::query(&query);
        if let Some(config) = config {
            if let Some(thread_id) = config
                .get("configurable")
                .and_then(|c| c.get("thread_id"))
                .and_then(|v| v.as_str())
            {
                q = q.bind(thread_id);
            }
            if let Some(checkpoint_ns) = config
                .get("configurable")
                .and_then(|c| c.get("checkpoint_ns"))
                .and_then(|v| v.as_str())
            {
                q = q.bind(checkpoint_ns);
            }
            if let Some(checkpoint_id) = get_checkpoint_id(config) {
                q = q.bind(checkpoint_id);
            }
        }
        if let Some(before) = before {
            if let Some(before_id) = get_checkpoint_id(before) {
                q = q.bind(before_id);
            }
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| CheckpointError::Storage(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(Self::row_to_tuple(&row)?);
        }

        Ok(results)
    }
}

/// CheckpointError needs a Config variant for missing config fields.
/// We add it here since the base error type doesn't have one.
#[allow(dead_code)]
impl PostgresSaver {
    /// Wrap a config error message into a CheckpointError::Storage.
    fn config_error(msg: &str) -> CheckpointError {
        CheckpointError::Storage(format!("config error: {}", msg))
    }
}
