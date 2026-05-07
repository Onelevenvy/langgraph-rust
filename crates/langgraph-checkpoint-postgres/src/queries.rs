/// SQL migrations for the checkpoint schema.
pub const MIGRATIONS: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY);",
    "CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        parent_checkpoint_id TEXT,
        type TEXT,
        checkpoint JSONB NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    );",
    "CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        channel TEXT NOT NULL,
        version TEXT NOT NULL,
        type TEXT NOT NULL,
        blob BYTEA,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    );",
    "CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id TEXT NOT NULL,
        checkpoint_ns TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        idx INTEGER NOT NULL,
        channel TEXT NOT NULL,
        type TEXT,
        blob BYTEA NOT NULL,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    );",
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP NOT NULL;",
    "SELECT 1;",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);",
    "ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';",
];

/// Select a checkpoint with its channel values and pending writes.
pub const SELECT_SQL: &str = r#"
SELECT
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        SELECT array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        FROM jsonb_each_text(checkpoint -> 'channel_versions')
        INNER JOIN checkpoint_blobs bl
            ON bl.thread_id = checkpoints.thread_id
            AND bl.checkpoint_ns = checkpoints.checkpoint_ns
            AND bl.channel = jsonb_each_text.key
            AND bl.version = jsonb_each_text.value
    ) AS channel_values,
    (
        SELECT array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] ORDER BY cw.task_id, cw.idx)
        FROM checkpoint_writes cw
        WHERE cw.thread_id = checkpoints.thread_id
            AND cw.checkpoint_ns = checkpoints.checkpoint_ns
            AND cw.checkpoint_id = checkpoints.checkpoint_id
    ) AS pending_writes
FROM checkpoints
"#;

/// Upsert checkpoint blobs.
pub const UPSERT_CHECKPOINT_BLOBS_SQL: &str = r#"
INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING
"#;

/// Upsert a checkpoint.
pub const UPSERT_CHECKPOINTS_SQL: &str = r#"
INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
DO UPDATE SET
    checkpoint = EXCLUDED.checkpoint,
    metadata = EXCLUDED.metadata
"#;

/// Upsert checkpoint writes (overwrite on conflict).
pub const UPSERT_CHECKPOINT_WRITES_SQL: &str = r#"
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO UPDATE SET
    channel = EXCLUDED.channel,
    type = EXCLUDED.type,
    blob = EXCLUDED.blob
"#;

/// Insert checkpoint writes (ignore on conflict).
pub const INSERT_CHECKPOINT_WRITES_SQL: &str = r#"
INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
"#;

/// Select pending sends for migration.
pub const SELECT_PENDING_SENDS_SQL: &str = r#"
SELECT
    checkpoint_id,
    array_agg(array[type::bytea, blob] ORDER BY task_path, task_id, idx) AS sends
FROM checkpoint_writes
WHERE thread_id = $1
    AND checkpoint_id = ANY($2)
    AND channel = '__tasks__'
GROUP BY checkpoint_id
"#;
