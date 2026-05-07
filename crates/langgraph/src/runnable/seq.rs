use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use super::base::{Runnable, RunnableError};

/// Chains multiple `Runnable`s sequentially — output of step N becomes input of step N+1.
///
/// This is the Rust equivalent of Python's `RunnableSeq` / `RunnableSequence`.
/// Requires at least 2 steps (first = node logic, rest = writers/post-processors).
///
/// In LangGraph's Pregel engine, every node is assembled as:
/// `RunnableSeq(node_func, channel_write)`
pub struct RunnableSeq {
    name: String,
    steps: Vec<Arc<dyn Runnable>>,
}

impl RunnableSeq {
    /// Create a new sequence. Panics if fewer than 2 steps.
    pub fn new(name: impl Into<String>, steps: Vec<Arc<dyn Runnable>>) -> Self {
        assert!(steps.len() >= 2, "RunnableSeq requires at least 2 steps");
        Self {
            name: name.into(),
            steps,
        }
    }

    /// Create with any number of steps (1 is allowed for degenerate cases).
    pub fn new_relaxed(name: impl Into<String>, steps: Vec<Arc<dyn Runnable>>) -> Self {
        Self {
            name: name.into(),
            steps,
        }
    }

    /// Number of steps in the sequence.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[async_trait]
impl Runnable for RunnableSeq {
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let mut current = input.clone();
        for step in &self.steps {
            current = step.invoke(&current, config)?;
        }
        Ok(current)
    }

    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let mut current = input.clone();
        for step in &self.steps {
            current = step.ainvoke(&current, config).await?;
        }
        Ok(current)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Pipe operator support: `a | b` creates `RunnableSeq("a|b", [a, b])`.
///
/// Since Rust doesn't support operator overloading on trait objects,
/// use the `pipe` free function instead:
/// ```ignore
/// let seq = pipe(a, b); // equivalent to a | b in Python
/// ```
pub fn pipe(first: Arc<dyn Runnable>, second: Arc<dyn Runnable>) -> RunnableSeq {
    let name = format!("{}|{}", first.name(), second.name());
    RunnableSeq::new_relaxed(name, vec![first, second])
}
