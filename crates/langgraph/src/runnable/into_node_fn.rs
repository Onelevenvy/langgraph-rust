use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;

use super::base::{Runnable, RunnableError};
use super::callable::RunnableCallable;

/// Type alias for the boxed future returned by async node functions.
///
/// Use this in return-position of factory functions that produce node closures:
/// ```ignore
/// fn my_node(model: Arc<dyn BaseChatModel>) -> impl Fn(JsonValue, RunnableConfig) -> NodeFnFuture + Send + Sync + 'static {
///     move |input, config| {
///         let model = model.clone();
///         Box::pin(async move { ... })
///     }
/// }
/// ```
pub type NodeFnFuture = Pin<Box<dyn Future<Output = Result<JsonValue, RunnableError>> + Send>>;

/// Trait for converting closures into node Runnables.
///
/// This enables `add_node` and `add_conditional_edges` to accept both
/// async and sync closures with a uniform API.
///
/// Implemented for:
/// - Async closures: `Fn(JsonValue, RunnableConfig) -> impl Future<...>`
/// - Via `SyncNodeFn` wrapper: sync closures
/// - `Arc<dyn Runnable>`: pre-built runnables
pub trait IntoNodeFunction {
    fn into_runnable(self, name: &str) -> Arc<dyn Runnable>;
}

// Blanket impl for async closures
impl<F, Fut> IntoNodeFunction for F
where
    F: Fn(JsonValue, RunnableConfig) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<JsonValue, RunnableError>> + Send + 'static,
{
    fn into_runnable(self, name: &str) -> Arc<dyn Runnable> {
        Arc::new(RunnableCallable::new(name, self))
    }
}

/// Wrapper for sync node functions.
///
/// Use this to pass sync closures to `add_node`:
/// ```ignore
/// graph.add_node("my_node", SyncNodeFn(|input, _config| {
///     Ok(json!({"result": 42}))
/// }));
/// ```
pub struct SyncNodeFn<F>(pub F);

impl<F> IntoNodeFunction for SyncNodeFn<F>
where
    F: Fn(&JsonValue, &RunnableConfig) -> Result<JsonValue, RunnableError>
        + Send + Sync + 'static,
{
    fn into_runnable(self, name: &str) -> Arc<dyn Runnable> {
        Arc::new(RunnableCallable::new_sync(name, self.0))
    }
}

/// Implement `IntoNodeFunction` for pre-built runnables.
impl IntoNodeFunction for Arc<dyn Runnable> {
    fn into_runnable(self, _name: &str) -> Arc<dyn Runnable> {
        self
    }
}

/// Wrapper for single-argument sync node functions that ignore config.
///
/// ```ignore
/// use langgraph::prelude::*;
///
/// graph.add_node("doubler", NodeFn1(|input| {
///     let n = input.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
///     Ok(json!({"value": n * 2}))
/// }));
/// ```
pub struct NodeFn1<F>(pub F);

impl<F> IntoNodeFunction for NodeFn1<F>
where
    F: Fn(&JsonValue) -> Result<JsonValue, RunnableError> + Send + Sync + 'static,
{
    fn into_runnable(self, name: &str) -> Arc<dyn Runnable> {
        let f = self.0;
        Arc::new(RunnableCallable::new_sync(name, move |input: &JsonValue, _config: &RunnableConfig| {
            f(input)
        }))
    }
}

/// Wrapper for routing functions used with `add_conditional_edges`.
///
/// Wraps a function `Fn(&JsonValue) -> String` so it can be used
/// directly as the `path` argument to `add_conditional_edges`.
///
/// ```ignore
/// use langgraph::prelude::*;
/// use langgraph_prebuilt::tools_condition;
///
/// graph.add_conditional_edges(
///     "agent",
///     RoutingFn(tools_condition),
///     Some(HashMap::from([
///         ("tools".to_string(), "tools".to_string()),
///         (END.to_string(), END.to_string()),
///     ])),
/// )?;
/// ```
pub struct RoutingFn<F>(pub F);

impl<F> IntoNodeFunction for RoutingFn<F>
where
    F: Fn(&JsonValue) -> String + Send + Sync + 'static,
{
    fn into_runnable(self, name: &str) -> Arc<dyn Runnable> {
        let f = self.0;
        Arc::new(RunnableCallable::new_sync(name, move |input: &JsonValue, _config: &RunnableConfig| {
            let route = f(input);
            Ok(JsonValue::String(route))
        }))
    }
}

/// Convenience macro to wrap a sync closure for use with `add_node`.
///
/// ```ignore
/// use langgraph::prelude::*;
///
/// graph.add_node("doubler", node_fn!(|input, _config| {
///     let n = input.as_i64().unwrap_or(0);
///     Ok(json!(n * 2))
/// }));
/// ```
#[macro_export]
macro_rules! node_fn {
    ($f:expr) => {
        $crate::runnable::SyncNodeFn($f)
    };
}

/// Wrap a routing function for use with `add_conditional_edges`.
///
/// ```ignore
/// use langgraph::prelude::*;
///
/// // Instead of: RoutingFn(tools_condition)
/// graph.add_conditional_edges("agent", routing!(tools_condition), None)?;
/// ```
#[macro_export]
macro_rules! routing {
    ($f:expr) => {
        $crate::runnable::RoutingFn($f)
    };
}

/// Simplify `add_conditional_edges` with automatic route map construction.
///
/// Python:
/// ```python
/// graph.add_conditional_edges("chatbot", tools_condition)
/// ```
///
/// Rust (before):
/// ```ignore
/// graph.add_conditional_edges(
///     "chatbot",
///     RoutingFn(tools_condition),
///     Some({
///         let mut map = HashMap::new();
///         map.insert("tools".to_string(), "tools".to_string());
///         map.insert(END.to_string(), END.to_string());
///         map
///     }),
/// )?;
/// ```
///
/// Rust (after):
/// ```ignore
/// conditional_edges!(graph, "chatbot", tools_condition, "tools" => "tools", END => END)?;
/// ```
#[macro_export]
macro_rules! conditional_edges {
    ($graph:expr, $source:expr, $route_fn:expr, $($key:expr => $val:expr),+ $(,)?) => {
        $graph.add_conditional_edges(
            $source,
            $crate::runnable::RoutingFn($route_fn),
            Some({
                let mut map = std::collections::HashMap::new();
                $(map.insert($key.to_string(), $val.to_string());)+
                map
            }),
        )
    };
}
