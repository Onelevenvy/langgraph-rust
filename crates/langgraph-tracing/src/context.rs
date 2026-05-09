use crate::event_bus::EventBus;
use crate::observer::TracingGraphObserver;
use crate::store::InMemoryTracingStore;
use crate::types::TraceStatus;
use langgraph_checkpoint::config::RunnableConfig;
use serde_json::Value as JsonValue;
use std::sync::Arc;

/// High-level tracing context for LangGraph applications.
///
/// Bundles store, event bus, and observer into a single convenient API.
/// Use with `#[derive(Traceable)]` on your state struct for automatic setup.
pub struct TracingContext {
    store: Arc<InMemoryTracingStore>,
    event_bus: EventBus,
    observer: TracingGraphObserver,
}

impl TracingContext {
    /// Access the underlying observer for low-level tracing operations.
    pub fn observer(&mut self) -> &mut TracingGraphObserver {
        &mut self.observer
    }

    /// Access the underlying store.
    pub fn store(&self) -> &Arc<InMemoryTracingStore> {
        &self.store
    }

    /// Access the underlying event bus.
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }

    /// Create a new tracing context with in-memory store.
    pub fn new() -> Self {
        let store = Arc::new(InMemoryTracingStore::new());
        let event_bus = EventBus::new();
        let observer = TracingGraphObserver::new(store.clone(), event_bus.clone());
        Self {
            store,
            event_bus,
            observer,
        }
    }

    /// Start the tracing web server in a background task.
    pub fn start_server(&self, addr: &str) {
        let store = self.store.clone();
        let event_bus = self.event_bus.clone();
        let addr = addr.to_string();
        tokio::spawn(async move {
            crate::server::start(
                &addr,
                store,
                event_bus,
                Some("crates/langgraph-tracing/frontend/dist"),
            )
            .await
            .unwrap();
        });
    }

    /// Run a graph with automatic tracing.
    ///
    /// The closure receives a `RunnableConfig` (with `trace_id` injected into
    /// `configurable`) and should execute the graph, returning the output.
    ///
    /// `on_graph_start` and `on_graph_end` are called automatically.
    pub async fn run_with_tracing<F, Fut>(
        &mut self,
        name: &str,
        input: JsonValue,
        base_config: RunnableConfig,
        f: F,
    ) -> JsonValue
    where
        F: FnOnce(RunnableConfig) -> Fut,
        Fut: std::future::Future<Output = JsonValue>,
    {
        let trace_id = self.observer.on_graph_start(name, input);

        // Merge trace_id into the base config's configurable
        let mut config = base_config;
        let configurable = config
            .entry("configurable".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(obj) = configurable.as_object_mut() {
            obj.insert("trace_id".to_string(), serde_json::json!(trace_id));
        }

        let output = f(config).await;

        self.observer
            .on_graph_end(&trace_id, output.clone(), TraceStatus::Success);
        output
    }
}

impl Default for TracingContext {
    fn default() -> Self {
        Self::new()
    }
}
