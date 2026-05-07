use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use crate::config::RunnableConfig;

/// Channel version can be a string, int, or float
pub type ChannelVersion = JsonValue;

/// Map of channel name -> version
pub type ChannelVersions = HashMap<String, ChannelVersion>;

/// Pending write: (task_id, channel, value)
pub type PendingWrite = (String, String, JsonValue);

/// Checkpoint format version
pub const LATEST_VERSION: i64 = 2;

/// Checkpoint source types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointSource {
    Input,
    Loop,
    Update,
    Fork,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<CheckpointSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parents: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

/// A checkpoint containing channel state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Format version
    pub v: i64,
    /// Monotonically increasing ID (UUIDv6)
    pub id: String,
    /// ISO 8601 timestamp
    pub ts: String,
    /// Channel name -> serialized value
    pub channel_values: HashMap<String, JsonValue>,
    /// Channel name -> version
    pub channel_versions: ChannelVersions,
    /// Node name -> (channel name -> last seen version)
    pub versions_seen: HashMap<String, ChannelVersions>,
    /// Channels updated in this checkpoint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_channels: Option<Vec<String>>,
}

/// A tuple grouping config + checkpoint + metadata + optional parent + pending writes
#[derive(Debug, Clone)]
pub struct CheckpointTuple {
    pub config: RunnableConfig,
    pub checkpoint: Checkpoint,
    pub metadata: CheckpointMetadata,
    pub parent_config: Option<RunnableConfig>,
    pub pending_writes: Option<Vec<PendingWrite>>,
}

impl Checkpoint {
    /// Create an empty checkpoint
    pub fn empty() -> Self {
        use chrono::Utc;
        use crate::checkpoint::id::uuid6;

        Self {
            v: LATEST_VERSION,
            id: uuid6(),
            ts: Utc::now().to_rfc3339(),
            channel_values: HashMap::new(),
            channel_versions: HashMap::new(),
            versions_seen: HashMap::new(),
            updated_channels: None,
        }
    }

    /// Deep copy a checkpoint
    pub fn copy(&self) -> Self {
        Self {
            v: self.v,
            id: self.id.clone(),
            ts: self.ts.clone(),
            channel_values: self.channel_values.clone(),
            channel_versions: self.channel_versions.clone(),
            versions_seen: self.versions_seen.clone(),
            updated_channels: self.updated_channels.clone(),
        }
    }
}
