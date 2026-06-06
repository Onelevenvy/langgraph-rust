use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use super::base::{Runnable, RunnableError};

/// A clonable async function: (input, config) -> future(result).
///
/// Uses `Arc` so the function can be cloned and shared across threads.
pub type BoxedFn = Arc<
    dyn Fn(
            JsonValue,
            RunnableConfig,
        ) -> Pin<Box<dyn Future<Output = Result<JsonValue, RunnableError>> + Send>>
        + Send
        + Sync,
>;

/// Wraps a function as a `Runnable`.
///
/// This is the Rust equivalent of Python's `RunnableCallable`.
/// Functions always receive `(JsonValue, RunnableConfig)` — the caller
/// is responsible for marshalling state into `JsonValue`.
pub struct RunnableCallable {
    name: String,
    func: BoxedFn,
}

impl RunnableCallable {
    /// Create from an async function.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(JsonValue, RunnableConfig) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<JsonValue, RunnableError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            func: Arc::new(move |input, config| Box::pin(f(input, config))),
        }
    }

    /// Create from a sync function (wrapped as async).
    pub fn new_sync<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&JsonValue, &RunnableConfig) -> Result<JsonValue, RunnableError> + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        Self {
            name: name.into(),
            func: Arc::new(move |input, config| {
                let f = f.clone();
                Box::pin(async move { f(&input, &config) })
            }),
        }
    }
}

#[async_trait]
impl Runnable for RunnableCallable {
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let func = self.func.clone();
        let input = input.clone();
        let config = config.clone();

        // Try to use existing tokio runtime, otherwise create one
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle.block_on(crate::config::with_config(config.clone(), func(input, config))),
            Err(_) => tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(func(input, config)),
        }
    }

    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let func = self.func.clone();
        let input = input.clone();
        let config_inner = config.clone();
        crate::config::with_config(config.clone(), func(input, config_inner)).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}
