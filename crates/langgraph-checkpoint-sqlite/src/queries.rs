//! SQL schema and statement constants for the SQLite checkpoint saver.
//!
//! SQLite-specific notes:
//! - Uses `TEXT` for JSON columns (SQLite has no native JSONB; `json_extract()`
//!   handles filtering).
//! - Uses `BLOB` for binary payloads (Postgres `BYTEA`).
//! - Bind parameters use `?` placeholders.
//! - WAL mode is enabled at connection setup for concurrent reads.

/// SQL migrations for the checkpoint schema. Applied in order; each
/// migration's index is recorded in `checkpoint_migrations.v`.
pub const MIGRATIONS: &[&str] = &[
    // 0: bootstrap migrations table
    "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY);",
    // 1: checkpoints table
    "CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type TEXT,
        checkpoint TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    );",
    // 2: checkpoint blobs (channel value storage)
    "CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        version TEXT NOT NULL,
        type TEXT NOT NULL,
        blob BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    );",
    // 3: checkpoint writes (pending task outputs)
    "CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        task_path TEXT NOT NULL DEFAULT '',
        idx INTEGER NOT NULL,
        channel TEXT NOT NULL,
        type TEXT,
        blob BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    );",
    // 4: indexes (CONCURRENTLY is Postgres-only; SQLite uses plain CREATE INDEX)
    "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);",
    // 5
    "CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);",
    // 6
    "CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);",
];

/// Select base columns for a checkpoint row. Channel values and pending
/// writes are fetched separately (SQLite lacks `array_agg`).
pub const SELECT_CHECKPOINT_SQL: &str = r#"
SELECT
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    type,
    checkpoint,
    metadata
FROM checkpoints
"#;

/// Fetch all blobs (channel values) for a given checkpoint by joining
/// `checkpoint.channel_versions` with the blobs table.
///
/// SQLite stores `version` as TEXT, while `je.value` from `json_each` may
/// be a JSON number or string depending on how the checkpoint was
/// produced. The explicit `CAST(... AS TEXT)` normalizes both sides so
/// integer and string versions compare equal.
pub const SELECT_BLOBS_SQL: &str = r#"
SELECT bl.channel, bl.type, bl.blob
FROM checkpoints cp
CROSS JOIN json_each(json_extract(cp.checkpoint, '$.channel_versions')) je
INNER JOIN checkpoint_blobs bl
    ON bl.thread_id = cp.thread_id
    AND bl.checkpoint_ns = cp.checkpoint_ns
    AND bl.channel = je.key
    AND bl.version = CAST(je.value AS TEXT)
WHERE cp.thread_id = ?1
  AND cp.checkpoint_ns = ?2
  AND cp.checkpoint_id = ?3
"#;

/// Fetch pending writes for a given checkpoint, ordered by task and idx.
pub const SELECT_WRITES_SQL: &str = r#"
SELECT task_id, channel, type, blob, idx, task_path
FROM checkpoint_writes
WHERE thread_id = ?1 AND checkpoint_ns = ?2 AND checkpoint_id = ?3
ORDER BY task_path ASC, task_id ASC, idx ASC
"#;

/// Upsert checkpoint blobs. Blobs are immutable per (channel, version),
/// so on conflict we keep the existing row.
pub const UPSERT_CHECKPOINT_BLOBS_SQL: &str = r#"
INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"#;

/// Upsert a checkpoint, replacing the JSON payload and metadata if the
/// (thread, ns, id) tuple already exists.
pub const UPSERT_CHECKPOINTS_SQL: &str = r#"
INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
VALUES (?1, ?2, ?3, ?4, ?5, ?6)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
    checkpoint = excluded.checkpoint,
    metadata = excluded.metadata
"#;

/// Upsert checkpoint writes (overwrite on conflict). Used for special
/// channels like `__error__` whose value may legitimately change.
pub const UPSERT_CHECKPOINT_WRITES_SQL: &str = r#"
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
    channel = excluded.channel,
    type = excluded.type,
    blob = excluded.blob
"#;

/// Insert checkpoint writes (ignore on conflict). Default for normal
/// per-task writes which should be append-only.
pub const INSERT_CHECKPOINT_WRITES_SQL: &str = r#"
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"#;
