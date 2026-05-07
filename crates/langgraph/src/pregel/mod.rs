pub mod algo;
pub mod read;
pub mod write;
pub mod runner;
pub mod io;

pub use algo::{prepare_next_tasks, apply_writes};
pub use read::PregelNode;
pub use write::{ChannelWrite, ChannelWriteEntry};
pub use runner::PregelRunner;
pub use io::{map_input, map_command, read_channels, NULL_TASK_ID};

use std::collections::HashMap;
use std::sync::Arc;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use crate::channels::Channel;
use crate::runnable::Runnable;

/// A task ready for execution in the current super-step.
#[derive(Debug, Clone)]
pub struct PregelTask {
    /// Name of the node to execute.
    pub name: String,
    /// Input value for this task.
    pub input: JsonValue,
    /// Channels this task reads from.
    pub input_channels: Vec<String>,
    /// Channels that triggered this task.
    pub triggers: Vec<String>,
    /// Unique task ID (deterministic from checkpoint + step + name).
    pub id: String,
}

/// An executable task with its runnable and write buffer.
pub struct PregelExecutableTask {
    /// Name of the node.
    pub name: String,
    /// Input value.
    pub input: JsonValue,
    /// The runnable to execute (node logic + writers).
    pub proc: Arc<dyn Runnable>,
    /// Write buffer: (channel, value) pairs collected during execution.
    pub writes: Vec<(String, JsonValue)>,
    /// Task configuration (with CONFIG_KEY_SEND, CONFIG_KEY_READ, etc.).
    pub config: RunnableConfig,
    /// Channels that triggered this task.
    pub triggers: Vec<String>,
    /// Unique task ID.
    pub id: String,
}

/// Status of the Pregel loop.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopStatus {
    /// Processing initial input.
    Input,
    /// Tasks pending execution.
    Pending,
    /// All done — no more tasks.
    Done,
    /// Interrupted before node execution.
    InterruptBefore,
    /// Interrupted after node execution.
    InterruptAfter,
    /// Step limit reached.
    OutOfSteps,
}

/// Channel version tracking.
pub type ChannelVersions = HashMap<String, JsonValue>;

/// Reverse index from channel name to nodes triggered by that channel.
pub type TriggerToNodes = HashMap<String, Vec<String>>;

/// Build the reverse index: channel_name -> [node_names that are triggered by it].
pub fn build_trigger_to_nodes(nodes: &HashMap<String, PregelNode>) -> TriggerToNodes {
    let mut map: TriggerToNodes = HashMap::new();
    for (name, node) in nodes {
        for trigger in &node.triggers {
            map.entry(trigger.clone()).or_default().push(name.clone());
        }
    }
    map
}

/// Reconstruct live channels from a checkpoint.
pub fn channels_from_checkpoint(
    specs: &HashMap<String, Box<dyn Channel>>,
    checkpoint_channels: &HashMap<String, Option<JsonValue>>,
) -> HashMap<String, Box<dyn Channel>> {
    let mut channels = HashMap::new();
    for (key, spec) in specs {
        let cp = checkpoint_channels.get(key).and_then(|v| v.as_ref());
        channels.insert(key.clone(), spec.from_checkpoint(cp));
    }
    channels
}
