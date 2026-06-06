# langgraph-rust

A Rust implementation of [LangGraph](https://github.com/langchain-ai/langgraph) -- a framework for building stateful, multi-actor applications with LLMs.

## Projects Built with langgraph-rust
> [!TIP]
> **Featured Project**: Check out [flock](https://github.com/Onelevenvy/flock) [![GitHub stars](https://img.shields.io/github/stars/Onelevenvy/flock?style=social)](https://github.com/Onelevenvy/flock) — a premium multi-agent framework built on top of langgraph-rust!

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Supported Checkpointers](#supported-checkpointers)
- [Roadmap](#roadmap)
- [Examples](#examples)
- [Crate Overview](#crate-overview)
- [Requirements](#requirements)
- [License](#license)

## Overview

langgraph-rust brings the core LangGraph concepts into idiomatic Rust:

- **StateGraph** -- Directed graphs where nodes are async functions that transform state, and edges define control flow
- **Pregel Engine** -- Bulk Synchronous Parallel (BSP) execution: Plan -> Execute (parallel via tokio) -> Update
- **Channels** -- Type-erased state containers with reducers, fan-in barriers, and pub-sub semantics
- **Checkpointing** -- Full state persistence for pause/resume, human-in-the-loop, and time-travel debugging. Supports **InMemory**, **Postgres**, and **SQLite**.
- **Tracing & Observability** -- Real-time tracing server and UI for visualizing graph execution and LLM calls.
- **ReAct Agent** -- Prebuilt Reasoning + Acting agent pattern with tool execution.
- **OpenAI Provider** -- Integration with OpenAI-compatible APIs (GPT-4o, Ollama, vLLM, DeepSeek, etc.).
## Project Structure

```
langgraph-rust/
├── crates/
│   ├── langgraph/                 # Core engine: StateGraph, Pregel BSP, Channels, Streaming
│   ├── langgraph-derive/          # Proc macros: #[derive(StateGraph)], #[tool], #[derive(Traceable)]
│   ├── langgraph-prebuilt/        # Prebuilt components: ReAct agent, ToolNode, Messages
│   ├── langgraph-checkpoint/      # Checkpointer traits and InMemorySaver
│   ├── langgraph-checkpoint-postgres/  # Postgres persistence via sqlx
│   ├── langgraph-checkpoint-sqlite/    # SQLite persistence via sqlx
│   ├── langgraph-tracing/         # Real-time tracing server and observability
│   ├── langgraph-providers/       # LLM integrations: OpenAI, OpenAI-compatible
│   └── langgraph-prebuilt/        # Prebuilt agents and nodes
└── examples/                      # 16 runnable examples covering all features
```

## Quick Start

### Installation

Add `langgraph` to your project:

```bash
# Add basic core package (no optional features by default)
cargo add langgraph

# Or enable specific features (e.g. prebuilt agents, model providers, sqlite checkpointer)
cargo add langgraph --features prebuilt,providers,sqlite,postgres

# Or enable all features at once
cargo add langgraph --features full
```

Or configure it in your `Cargo.toml` manually:

```toml
[dependencies]
# Basic core package
langgraph = "0.2.1"

# Or enable specific features:
# langgraph = { version = "0.2.1", features = ["prebuilt", "providers", "sqlite", "postgres"] }

# Or enable all features at once:
# langgraph = { version = "0.2.1", features = ["full"] }
```

### Cargo Features

The `langgraph` umbrella crate organizes the ecosystem into separate cargo features:

- `full`: Enables all optional features below (`prebuilt`, `providers`, `tracing`, `sqlite`, `postgres`).
- `prebuilt`: ReAct agent, `ToolNode`, message types, and base chat model trait.
- `providers`: Model providers integration (OpenAI, compatibility clients).
- `tracing`: Tracing server, observer, and telemetry.
- `sqlite`: SQLite checkpointer persistence.
- `postgres`: PostgreSQL checkpointer persistence.


### Configure Environment

Copy `.env.example` to `.env` and fill in your specific model information:

```bash
cp .env.example .env
```

Example `.env` content:
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_BASE=https://xxxxxxx/v1
OPENAI_MODEL=xxxxxxx
```

### Define a ReAct Agent with Tools

```rust
use langgraph_derive::tool;
use langgraph_prebuilt::{create_react_agent, prepare_tools, ReActAgentConfig};
use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
use langgraph_checkpoint::config::RunnableConfig;
use std::sync::Arc;

// Define tools with the #[tool] macro
#[tool("get_weather", "Get the current weather for a given location.")]
fn get_weather(location: String) -> Result<String, String> {
    Ok(format!("Weather for {}: sunny, 22°C", location))
}

#[tool("calculator", "Evaluate a mathematical expression.")]
fn calculator(expression: String) -> Result<String, String> {
    Ok(format!("{} = 42", expression))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prepare tools
    let prepared = prepare_tools(vec![
        Arc::new(GetWeather::new()),
        Arc::new(Calculator::new()),
    ]);

    // Create LLM
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: "gpt-4o".to_string(),
        api_key: std::env::var("OPENAI_API_KEY")?,
        api_base: None,
        temperature: Some(0.7),
        ..Default::default()
    });

    // Build ReAct agent
    let agent = create_react_agent(
        Arc::new(model),
        prepared.tools,
        Some(ReActAgentConfig {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            max_steps: Some(10),
            handle_tool_errors: true,
        }),
    )?;

    // Invoke
    let input = serde_json::json!({
        "messages": [{"type": "human", "content": "What's the weather in Beijing?"}]
    });
    let result = agent.ainvoke(&input, &RunnableConfig::new()).await?;
    println!("{:#?}", result);

    Ok(())
}
```

### Build a Custom StateGraph

```rust
use langgraph::prelude::*;
use langgraph_derive::langgraph_state;
use langgraph::channels::binop::append_reducer;

#[langgraph_state]
#[derive(Debug)]
struct MyState {
    #[channel(reducer = "append_reducer")]
    steps: Vec<String>,
    result: String,
}

// Build graph: START -> process -> validate -> END
let channels = MyState::create_channels();
let mut graph = StateGraph::new(channels);

graph.add_node("process", node_fn!(|input, _config| {
    Ok(serde_json::json!({
        "steps": vec!["processed"],
        "result": "done"
    }))
}))?;

graph.add_node("validate", node_fn!(|input, _config| {
    Ok(serde_json::json!({
        "steps": vec!["validated"]
    }))
}))?;

graph.add_edge(START, "process")?;
graph.add_edge("process", "validate")?;
graph.add_edge("validate", END)?;
let app = graph.compile()?;
```

### Human-in-the-Loop

```rust
use langgraph::types::{interrupt, Command};

// In a node function:
fn approval_node(state: &MyState) -> MyState {
    let human_response = interrupt("Please review and approve");
    // Execution pauses here, resumes when Command::resume is provided
    state.clone()
}

// Resume execution:
compiled_graph.update_state(
    &config,
    &Command::resume("approved".into()),
);
```

## Supported Providers

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | Done | GPT-4o, GPT-4, GPT-3.5, etc. via `async-openai` |
| OpenAI-compatible | Done | Ollama, vLLM, LiteLLM, Azure OpenAI via custom `base_url` |

## Supported Checkpointers

| Checkpointer | Status | Notes |
|--------------|--------|-------|
| InMemorySaver | Done | HashMap-based, for testing and development |
| PostgresSaver | Done | Production-ready via `sqlx`, with migrations |
| SqliteSaver | Done | Lightweight persistence via `sqlx` (SQLite) |

## Roadmap

### Providers

- [x] Anthropic (Claude)
- [ ] DeepSeek
- [ ] Google Gemini
- [ ] Qwen 
- [ ] Zhipu 

### Checkpointers

- [x] SQLite Checkpointer
- [ ] Redis Checkpointer


### Features


- [ ] Subgraph support
- [ ] More prebuilt agent patterns 

## Examples

| Example | Description |
|---------|-------------|
| `react_agent` | ReAct agent with `#[tool]` macro and `create_react_agent` |
| `interactive_chat` | Interactive CLI chat with memory and history |
| `interactive_chat_with_tracing` | Interactive chat with real-time tracing UI |
| `sqlite_checkpoint` | Using SQLite for state persistence |
| `human_in_the_loop` | `interrupt()` for human approval with `Command::resume` |
| `human_in_the_loop_sqlite_checkpoint` | HITL with SQLite storage |
| `streaming` | Token-by-token streaming with `StreamWriter` |
| `time_travel` | `get_state_history` and fork from checkpoint |
| `manus_like` | Plan-and-act multi-node agent (planner/executor/replanner) |
| `graph_with_tools` | Manual graph construction with tools and streaming |
| `state_graph_derive` | `#[derive(StateGraph)]` usage |
| `custom_state_hitl` | HITL with custom state and complex control flow |
| `parallel_interrupt_hitl` | Parallel execution with interrupts and HITL |
| `tracing_demos` | Demonstration of tracing capabilities |
| `langgraph_provider_openai` | Direct usage of the OpenAI provider |
| `join_edge_test` | Testing graph join edges and complex routing |

Run an example:

```bash
# Set your API key
export OPENAI_API_KEY=sk-xxx
# Optional: custom base URL for Ollama/vLLM
# export OPENAI_API_BASE=http://localhost:11434/v1

cargo run --example react_agent
cargo run --example interactive_chat
cargo run --example interactive_chat_with_tracing
cargo run --example human_in_the_loop
cargo run --example sqlite_checkpoint
```

## Crate Overview

| Crate | crates.io Crate Name | Description |
|-------|----------------------|-------------|
| `langgraph` | [**`langgraph`**](https://crates.io/crates/langgraph) | **Main Umbrella Crate** (Run `cargo add langgraph` to install the full suite) |
| `langgraph-core` | [**`langgraph-core-rs`**](https://crates.io/crates/langgraph-core-rs) | Core engine: StateGraph, Pregel BSP, Channels, Streaming, Runnable |
| `langgraph-derive` | [**`langgraph-derive`**](https://crates.io/crates/langgraph-derive) | `#[derive(StateGraph)]`, `#[tool]`, and `#[derive(Traceable)]` macros |
| `langgraph-prebuilt` | [**`langgraph-prebuilt`**](https://crates.io/crates/langgraph-prebuilt) | ReAct agent, ToolNode, Message types, BaseChatModel trait |
| `langgraph-checkpoint` | [**`langgraph-checkpoint-rs`**](https://crates.io/crates/langgraph-checkpoint-rs) | `BaseCheckpointSaver`, `InMemorySaver`, `InMemoryStore` |
| `langgraph-checkpoint-postgres` | [**`langgraph-checkpoint-postgres-rs`**](https://crates.io/crates/langgraph-checkpoint-postgres-rs) | `PostgresSaver` via sqlx with migrations |
| `langgraph-checkpoint-sqlite` | [**`langgraph-checkpoint-sqlite-rs`**](https://crates.io/crates/langgraph-checkpoint-sqlite-rs) | `SqliteSaver` via sqlx for SQLite |
| `langgraph-tracing` | [**`langgraph-tracing`**](https://crates.io/crates/langgraph-tracing) | Real-time tracing server, event bus, and observers |
| `langgraph-providers` | [**`langgraph-providers`**](https://crates.io/crates/langgraph-providers) | `OpenAIModel`, `OpenAICompatModel` (Ollama, vLLM, Azure, DeepSeek) |

## Requirements

- Rust >= 1.75
- Tokio runtime (full features)

## License

MIT
