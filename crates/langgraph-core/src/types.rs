use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use crate::config;
use crate::constants::{CONFIG_KEY_SCRATCHPAD, CONFIG_KEY_CHECKPOINT_NS};

/// Durability mode for checkpoint writes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    Sync,
    Async,
    Exit,
}

/// Stream mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum StreamMode {
    Values,
    Updates,
    Checkpoints,
    Tasks,
    Debug,
    Messages,
    Custom,
}

/// An interrupt value surfaced to the client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    /// Value surfaced to client
    pub value: JsonValue,
    /// Unique interrupt ID
    pub id: String,
}

/// Send a message to a target node (for map-reduce patterns)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Send {
    /// Target node name
    pub node: String,
    /// State/message to send
    pub arg: JsonValue,
}

impl Send {
    pub fn new(node: impl Into<String>, arg: JsonValue) -> Self {
        Self {
            node: node.into(),
            arg,
        }
    }
}

impl std::hash::Hash for Send {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        // Hash the JSON string representation
        self.arg.to_string().hash(state);
    }
}

impl PartialEq for Send {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.arg == other.arg
    }
}

impl Eq for Send {}

/// Command for controlling graph execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// None = current graph, "__parent__" = parent graph
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph: Option<String>,
    /// State update
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<JsonValue>,
    /// Resume value(s) for interrupt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<JsonValue>,
    /// Navigation targets
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub goto: Vec<CommandGoto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommandGoto {
    Node(String),
    Send(Send),
}

impl Command {
    pub const PARENT: &'static str = "__parent__";

    pub fn new() -> Self {
        Self {
            graph: None,
            update: None,
            resume: None,
            goto: Vec::new(),
        }
    }

    pub fn resume(value: JsonValue) -> Self {
        Self {
            resume: Some(value),
            ..Self::new()
        }
    }

    pub fn goto(node: impl Into<String>) -> Self {
        Self {
            goto: vec![CommandGoto::Node(node.into())],
            ..Self::new()
        }
    }
}

impl Default for Command {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry policy for task execution
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub initial_interval: f64,
    pub backoff_factor: f64,
    pub max_interval: f64,
    pub max_attempts: usize,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            initial_interval: 0.5,
            backoff_factor: 2.0,
            max_interval: 128.0,
            max_attempts: 3,
            jitter: true,
        }
    }
}

/// Cache policy for task results
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct CachePolicy {
    pub ttl: Option<i64>,
}


/// Overwrite bypasses the reducer and writes directly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Overwrite {
    pub value: JsonValue,
}

/// Lightweight task descriptor (for state queries)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PregelTask {
    pub id: String,
    pub name: String,
    pub path: Vec<String>,
    #[serde(skip)]
    pub error: Option<String>,
    pub interrupts: Vec<Interrupt>,
    #[serde(skip)]
    pub result: Option<JsonValue>,
}

/// The full task ready for execution
#[derive(Debug)]
pub struct PregelExecutableTask {
    pub name: String,
    pub input: JsonValue,
    pub writes: Vec<(String, JsonValue)>,
    pub config: RunnableConfig,
    pub triggers: Vec<String>,
    pub retry_policy: Vec<RetryPolicy>,
    pub id: String,
    pub path: Vec<String>,
}

/// State snapshot returned by get_state()
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub values: JsonValue,
    pub next: Vec<String>,
    pub config: RunnableConfig,
    pub metadata: Option<langgraph_checkpoint::CheckpointMetadata>,
    pub created_at: Option<String>,
    pub parent_config: Option<RunnableConfig>,
    pub tasks: Vec<PregelTask>,
    pub interrupts: Vec<Interrupt>,
}

/// Scratchpad for tracking interrupt state during execution.
///
/// This is stored in config["configurable"]["__pregel_scratchpad"]
/// and is used by the interrupt() function to manage resume values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PregelScratchpad {
    /// Current step number
    pub step: u64,
    /// Interrupt counter (incremented each time interrupt() is called)
    pub interrupt_counter: u64,
    /// Resume values from Command(resume=...)
    pub resume: Vec<JsonValue>,
    /// Whether we are resuming from an interrupt
    pub is_resuming: bool,
}

impl PregelScratchpad {
    pub fn new(step: u64) -> Self {
        Self {
            step,
            interrupt_counter: 0,
            resume: Vec::new(),
            is_resuming: false,
        }
    }

    /// Increment and return the interrupt counter
    pub fn next_interrupt_id(&mut self) -> u64 {
        let id = self.interrupt_counter;
        self.interrupt_counter += 1;
        id
    }
}

/// Error type for graph interrupts
#[derive(Debug, Clone)]
pub struct GraphInterrupt {
    pub interrupts: Vec<Interrupt>,
}

impl std::fmt::Display for GraphInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphInterrupt: {} interrupt(s)", self.interrupts.len())
    }
}

impl std::error::Error for GraphInterrupt {}

/// Error type returned by `interrupt()`.
///
/// This type is designed to work seamlessly with the `?` operator:
/// - In functions returning `Result<T, String>`: converts to a string error
/// - In the `#[tool]` macro: converts to `ToolError::Interrupt` to propagate correctly
///
/// This means `interrupt(...)? ` works in any tool function regardless of error type.
#[derive(Debug, Clone)]
pub struct InterruptError(pub GraphInterrupt);

impl std::fmt::Display for InterruptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InterruptError {}

impl From<GraphInterrupt> for InterruptError {
    fn from(interrupt: GraphInterrupt) -> Self {
        InterruptError(interrupt)
    }
}

impl From<InterruptError> for GraphInterrupt {
    fn from(e: InterruptError) -> GraphInterrupt {
        e.0
    }
}

/// Interrupt the current graph execution with a value.
///
/// This function should be called from within a node to pause execution
/// and wait for human input. When the graph is resumed with
/// `Command(resume=value)`, this function returns the resume value.
///
/// Returns `Ok(value)` if a resume value is available, or `Err(InterruptError)`
/// if no resume value is found (meaning execution should pause).
///
/// # Example
/// ```rust,ignore
/// use langgraph::types::interrupt;
///
/// #[tool("human_assistance", "Request help from a human")]
/// fn human_assistance(query: String) -> Result<String, String> {
///     let response = interrupt(json!({"query": query}))?;
///     Ok(response.to_string())
/// }
/// ```
pub fn interrupt(value: JsonValue) -> Result<JsonValue, InterruptError> {
    let config = config::get_config();

    // Check if there's a resume value in the scratchpad
    if let Some(configurable) = config.get("configurable") {
        if let Some(scratchpad_val) = configurable.get(CONFIG_KEY_SCRATCHPAD) {
            if let Ok(mut scratchpad) = serde_json::from_value::<PregelScratchpad>(scratchpad_val.clone()) {
                let idx = scratchpad.next_interrupt_id() as usize;

                // Check if we have a resume value for this interrupt
                if idx < scratchpad.resume.len() {
                    let resume_value = scratchpad.resume[idx].clone();
                    return Ok(resume_value);
                }
            }
        }
    }

    // No resume value available - raise GraphInterrupt
    Err(InterruptError(GraphInterrupt {
        interrupts: vec![Interrupt {
            value,
            id: uuid_from_config(&config),
        }],
    }))
}

/// Generate a deterministic UUID from config for interrupt IDs.
fn uuid_from_config(config: &HashMap<String, JsonValue>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let checkpoint_ns = config
        .get("configurable")
        .and_then(|c| c.get(CONFIG_KEY_CHECKPOINT_NS))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let mut hasher = DefaultHasher::new();
    checkpoint_ns.hash(&mut hasher);
    let hash = hasher.finish();

    format!("{:016x}", hash)
}
