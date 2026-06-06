pub mod base;
pub mod callable;
pub mod seq;
pub mod into_node_fn;

pub use base::{Runnable, RunnableError};
pub use callable::RunnableCallable;
pub use seq::{RunnableSeq, pipe};
pub use into_node_fn::{IntoNodeFunction, SyncNodeFn, NodeFnFuture, NodeFn1, RoutingFn};

use std::sync::Arc;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;

/// Coerce a closure into a `Runnable`.
///
/// In Python this handles: Runnable instances, callables, dicts (RunnableParallel).
/// In Rust, since closures aren't trait objects, this is the explicit conversion point.
///
/// Usage:
/// ```ignore
/// let r = coerce_to_runnable("my_fn", |input, config| async move {
///     Ok(json!({"result": input}))
/// });
/// ```
pub fn coerce_to_runnable<F, Fut>(name: impl Into<String>, f: F) -> Arc<dyn Runnable>
where
    F: Fn(JsonValue, RunnableConfig) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<JsonValue, RunnableError>> + Send + 'static,
{
    Arc::new(RunnableCallable::new(name, f))
}

/// Coerce a sync closure into a `Runnable`.
pub fn coerce_to_runnable_sync<F>(name: impl Into<String>, f: F) -> Arc<dyn Runnable>
where
    F: Fn(&JsonValue, &RunnableConfig) -> Result<JsonValue, RunnableError> + Send + Sync + 'static,
{
    Arc::new(RunnableCallable::new_sync(name, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callable_sync() {
        let r = RunnableCallable::new_sync("double", |input, _config| {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n * 2))
        });

        let result = r.invoke(&serde_json::json!(5), &RunnableConfig::new()).unwrap();
        assert_eq!(result, serde_json::json!(10));
        assert_eq!(r.name(), "double");
    }

    #[tokio::test]
    async fn test_callable_async() {
        let r = RunnableCallable::new("async_double", |input, _config| async move {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n * 2))
        });

        let result = r.ainvoke(&serde_json::json!(7), &RunnableConfig::new()).await.unwrap();
        assert_eq!(result, serde_json::json!(14));
    }

    #[tokio::test]
    async fn test_seq_chain() {
        let add_one = RunnableCallable::new("add_one", |input, _config| async move {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n + 1))
        });
        let double = RunnableCallable::new("double", |input, _config| async move {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n * 2))
        });

        let seq = RunnableSeq::new("add_then_double", vec![
            Arc::new(add_one) as Arc<dyn Runnable>,
            Arc::new(double),
        ]);

        // (5 + 1) * 2 = 12
        let result = seq.ainvoke(&serde_json::json!(5), &RunnableConfig::new()).await.unwrap();
        assert_eq!(result, serde_json::json!(12));
    }

    #[tokio::test]
    async fn test_coerce_to_runnable() {
        let r = coerce_to_runnable("echo", |input, _config| async move {
            Ok(input)
        });

        let result = r.ainvoke(&serde_json::json!({"hello": "world"}), &RunnableConfig::new()).await.unwrap();
        assert_eq!(result, serde_json::json!({"hello": "world"}));
    }

    #[test]
    fn test_seq_invoke_sync() {
        let add_one = RunnableCallable::new_sync("add_one", |input, _config| {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n + 1))
        });
        let double = RunnableCallable::new_sync("double", |input, _config| {
            let n = input.as_i64().unwrap_or(0);
            Ok(serde_json::json!(n * 2))
        });

        let seq = RunnableSeq::new("add_then_double", vec![
            Arc::new(add_one) as Arc<dyn Runnable>,
            Arc::new(double),
        ]);

        let result = seq.invoke(&serde_json::json!(3), &RunnableConfig::new()).unwrap();
        assert_eq!(result, serde_json::json!(8)); // (3+1)*2
    }

    #[test]
    fn test_pipe() {
        let a = Arc::new(RunnableCallable::new_sync("a", |input, _| Ok(input.clone()))) as Arc<dyn Runnable>;
        let b = Arc::new(RunnableCallable::new_sync("b", |input, _| Ok(input.clone()))) as Arc<dyn Runnable>;

        let seq = pipe(a, b);
        assert_eq!(seq.name(), "a|b");
        assert_eq!(seq.len(), 2);
    }
}
