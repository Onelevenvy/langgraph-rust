use crate::runtime::Runtime;
use langgraph_checkpoint::config::RunnableConfig;
use langgraph_checkpoint::store::base::BaseStore;
use serde_json::Value as JsonValue;
use std::cell::RefCell;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime as TokioRuntime;
use tokio::sync::mpsc;

// Task-local config for async contexts
tokio::task_local! {
    static ASYNC_CONFIG: RefCell<Option<RunnableConfig>>;
    static ASYNC_RUNTIME: RefCell<Option<Arc<Runtime>>>;
}

// Thread-local config for sync contexts
thread_local! {
    static SYNC_CONFIG: RefCell<Option<RunnableConfig>> = const { RefCell::new(None) };
    static SYNC_RUNTIME: RefCell<Option<Arc<Runtime>>> = const { RefCell::new(None) };
}

/// Get the current RunnableConfig from context.
///
/// Works in both sync and async contexts. Panics if called outside
/// a runnable execution context.
pub fn get_config() -> RunnableConfig {
    ASYNC_CONFIG
        .try_with(|c| c.borrow().clone())
        .ok()
        .flatten()
        .or_else(|| SYNC_CONFIG.with(|c| c.borrow().clone()))
        .expect("get_config() called outside of a runnable context")
}

/// Get the current Runtime from context.
pub fn get_runtime() -> Option<Arc<Runtime>> {
    ASYNC_RUNTIME
        .try_with(|r| r.borrow().clone())
        .ok()
        .flatten()
        .or_else(|| SYNC_RUNTIME.with(|r| r.borrow().clone()))
}

/// Get the current store from Runtime context.
pub fn get_store() -> Option<Arc<dyn BaseStore>> {
    get_runtime().and_then(|rt| rt.store.clone())
}

/// Get the StreamWriter from the current Runtime context, if available.
///
/// Returns `None` if not in a streaming context or if custom streaming
/// is not enabled. Nodes can use this to emit custom stream data:
///
/// ```ignore
/// use langgraph::config::get_stream_writer;
///
/// if let Some(writer) = get_stream_writer() {
///     let _ = writer.try_send(json!({"progress": 50}));
/// }
/// ```
pub fn get_stream_writer() -> Option<mpsc::Sender<JsonValue>> {
    get_runtime().and_then(|rt| rt.stream_writer.clone())
}

/// Set the config and runtime for the current async task scope.
pub async fn with_config<F, R>(config: RunnableConfig, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    ASYNC_CONFIG.scope(RefCell::new(Some(config)), f).await
}

/// Set the config, runtime, and store for the current async task scope.
pub async fn with_runtime<F, R>(config: RunnableConfig, runtime: Arc<Runtime>, f: F) -> R
where
    F: std::future::Future<Output = R>,
{
    ASYNC_CONFIG
        .scope(
            RefCell::new(Some(config)),
            ASYNC_RUNTIME.scope(RefCell::new(Some(runtime)), f),
        )
        .await
}

/// Set the config for the current sync scope.
pub fn with_config_sync<F, R>(config: RunnableConfig, f: F) -> R
where
    F: FnOnce() -> R,
{
    SYNC_CONFIG.with(|c| {
        *c.borrow_mut() = Some(config);
    });
    let result = f();
    SYNC_CONFIG.with(|c| {
        *c.borrow_mut() = None;
    });
    result
}

/// Set the config and runtime for the current sync scope.
pub fn with_runtime_sync<F, R>(config: RunnableConfig, runtime: Arc<Runtime>, f: F) -> R
where
    F: FnOnce() -> R,
{
    SYNC_CONFIG.with(|c| {
        *c.borrow_mut() = Some(config);
    });
    SYNC_RUNTIME.with(|r| {
        *r.borrow_mut() = Some(runtime);
    });
    let result = f();
    SYNC_CONFIG.with(|c| {
        *c.borrow_mut() = None;
    });
    SYNC_RUNTIME.with(|r| {
        *r.borrow_mut() = None;
    });
    result
}

/// Block on a future, reusing a process-global Tokio runtime when the caller
/// is outside a runtime.
///
/// The sync `invoke()` entry points used to build and drop a full multi-thread
/// `Runtime` on every call made outside a Tokio context — each construction
/// spawns `num_cpus` worker threads (~100µs+). A single cached runtime
/// amortizes that cost while preserving the exact scheduler semantics a
/// per-call `Runtime::new()` provided (multi-thread, IO + time enabled),
/// including the `block_in_place` bridge checkpoint savers rely on (it checks
/// `runtime_flavor() != CurrentThread`). When the caller is already inside a
/// runtime, `block_on` on its handle instead — no runtime is created or cached.
pub fn block_on<F>(fut: F) -> F::Output
where
    F: Future,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => handle.block_on(fut),
        Err(_) => CACHED_RUNTIME
            .get_or_init(|| {
                tokio::runtime::Runtime::new().expect("failed to build the shared Tokio runtime")
            })
            .block_on(fut),
    }
}

static CACHED_RUNTIME: OnceLock<TokioRuntime> = OnceLock::new();
