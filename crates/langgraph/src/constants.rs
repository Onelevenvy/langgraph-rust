/// Virtual start node
pub const START: &str = "__start__";

/// Virtual end (terminal) node
pub const END: &str = "__end__";

/// Disables streaming for a chat model
pub const TAG_NOSTREAM: &str = "nostream";

/// Hides node/edge from tracing
pub const TAG_HIDDEN: &str = "langsmith:hidden";

// Reserved write keys
/// Graph input
pub const INPUT: &str = "__input__";
/// Dynamic interrupts from nodes
pub const INTERRUPT: &str = "__interrupt__";
/// Values to resume after interrupt
pub const RESUME: &str = "__resume__";
/// Node errors
pub const ERROR: &str = "__error__";
/// Marker that node wrote nothing
pub const NO_WRITES: &str = "__no_writes__";
/// Channel for Send objects (PUSH tasks)
pub const TASKS: &str = "__pregel_tasks";
/// Records a task's return value
pub const RETURN: &str = "__return__";
/// Implicit branch for Control values
pub const PREVIOUS: &str = "__previous__";

// Task dispatch modes
/// Tasks created by Send objects
pub const PUSH: &str = "__pregel_push";
/// Tasks triggered by channel subscriptions/edges
pub const PULL: &str = "__pregel_pull";

// Namespace separators
/// Separates levels in checkpoint_ns (e.g., "graph|subgraph")
pub const NS_SEP: &str = "|";
/// Separates namespace from task_id within a level
pub const NS_END: &str = ":";

// Config keys (stored in config["configurable"])
pub const CONFIG_KEY_SEND: &str = "__pregel_send";
pub const CONFIG_KEY_READ: &str = "__pregel_read";
pub const CONFIG_KEY_CALL: &str = "__pregel_call";
pub const CONFIG_KEY_CHECKPOINTER: &str = "__pregel_checkpointer";
pub const CONFIG_KEY_STREAM: &str = "__pregel_stream";
pub const CONFIG_KEY_CACHE: &str = "__pregel_cache";
pub const CONFIG_KEY_RESUMING: &str = "__pregel_resuming";
pub const CONFIG_KEY_TASK_ID: &str = "__pregel_task_id";
pub const CONFIG_KEY_THREAD_ID: &str = "thread_id";
pub const CONFIG_KEY_CHECKPOINT_MAP: &str = "checkpoint_map";
pub const CONFIG_KEY_CHECKPOINT_ID: &str = "checkpoint_id";
pub const CONFIG_KEY_CHECKPOINT_NS: &str = "checkpoint_ns";
pub const CONFIG_KEY_SCRATCHPAD: &str = "__pregel_scratchpad";
pub const CONFIG_KEY_RUNNER_SUBMIT: &str = "__pregel_runner_submit";
pub const CONFIG_KEY_DURABILITY: &str = "__pregel_durability";
pub const CONFIG_KEY_RUNTIME: &str = "__pregel_runtime";
pub const CONFIG_KEY_RESUME_MAP: &str = "__pregel_resume_map";

/// Null task ID for writes not associated with a task
pub const NULL_TASK_ID: &str = "00000000-0000-0000-0000-000000000000";
