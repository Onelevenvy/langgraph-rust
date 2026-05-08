use crate::event_bus::EventBus;
use crate::store::{TraceFilter, TracingStore};
use crate::types::*;
use axum::extract::{Path, Query, State};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use serde::Deserialize;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Query parameters for listing traces
#[derive(Debug, Deserialize)]
struct ListTracesQuery {
    status: Option<String>,
    name: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

/// GET /api/traces
async fn list_traces(
    State(state): State<AppState>,
    Query(params): Query<ListTracesQuery>,
) -> Json<Vec<TraceSummary>> {
    let filter = TraceFilter {
        status: params.status.and_then(|s| match s.as_str() {
            "running" => Some(TraceStatus::Running),
            "success" => Some(TraceStatus::Success),
            "error" => Some(TraceStatus::Error),
            "interrupted" => Some(TraceStatus::Interrupted),
            _ => None,
        }),
        name_contains: params.name,
        limit: params.limit,
        offset: params.offset,
    };
    Json(state.store.list_traces(&filter))
}

/// GET /api/traces/:id
async fn get_trace(
    State(state): State<AppState>,
    Path(trace_id): Path<String>,
) -> Result<Json<TraceDetail>, axum::http::StatusCode> {
    state
        .store
        .get_trace(&trace_id)
        .map(Json)
        .ok_or(axum::http::StatusCode::NOT_FOUND)
}

/// DELETE /api/traces
async fn clear_traces(State(state): State<AppState>) {
    state.store.clear();
}

/// WebSocket handler for real-time events
async fn ws_handler(
    ws: axum::extract::WebSocketUpgrade,
    State(state): State<AppState>,
) -> axum::response::Response {
    ws.on_upgrade(|socket| handle_ws(socket, state))
}

async fn handle_ws(socket: axum::extract::ws::WebSocket, state: AppState) {
    let (mut sender, _receiver) = socket.split();
    let mut rx = state.event_bus.subscribe();

    while let Ok(event) = rx.recv().await {
        if let Ok(json) = serde_json::to_string(&event) {
            let msg: axum::extract::ws::Message = axum::extract::ws::Message::Text(json);
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    }
}

#[derive(Clone)]
struct AppState {
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
}

/// Start the tracing web server on the given address.
///
/// `static_dir` is the path to the built frontend assets (e.g., "crates/langgraph-tracing/frontend/dist").
/// If the directory doesn't exist, only the API and WebSocket endpoints will be available.
pub async fn start(
    addr: &str,
    store: Arc<dyn TracingStore>,
    event_bus: EventBus,
    static_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState { store, event_bus };

    let api_routes = Router::new()
        .route("/api/traces", get(list_traces).delete(clear_traces))
        .route("/api/traces/{trace_id}", get(get_trace))
        .route("/ws", get(ws_handler))
        .with_state(state);

    let app = if let Some(dir) = static_dir {
        let dir_owned = dir.to_string();
        let serve_dir = tower_http::services::ServeDir::new(&dir_owned)
            .not_found_service(tower_http::services::ServeDir::new(&dir_owned)
                .fallback(axum::routing::any(move || {
                    let d = dir_owned.clone();
                    async move {
                        match tokio::fs::read_to_string(format!("{}/index.html", d)).await {
                            Ok(html) => axum::response::Html(html).into_response(),
                            Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
                        }
                    }
                })));
        api_routes.merge(Router::new().fallback_service(serve_dir))
    } else {
        api_routes
    };

    let app = app.layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Tracing UI available at http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
