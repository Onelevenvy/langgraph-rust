//! Provider integrations for LangGraph.
//!
//! This crate provides concrete implementations of `BaseChatModel` and `BaseTool`
//! for popular LLM providers.
//!
//! # Supported Providers
//! - **OpenAI** — via `async-openai` crate (GPT-4o, GPT-4, o1, etc.)
//!
//! # Example
//! ```rust,no_run
//! use std::sync::Arc;
//! use langgraph_providers::openai::{OpenAIModel, OpenAIModelConfig};
//! use langgraph_prebuilt::{create_react_agent, ReActAgentConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = OpenAIModel::new(OpenAIModelConfig {
//!     model: "gpt-4o".to_string(),
//!     api_key: std::env::var("OPENAI_API_KEY")?,
//!     ..Default::default()
//! });
//!
//! let agent = create_react_agent(
//!     Arc::new(model),
//!     vec![], // tools
//!     Some(ReActAgentConfig {
//!         system_prompt: Some("You are a helpful assistant.".to_string()),
//!         ..Default::default()
//!     }),
//! )?;
//! # Ok(())
//! # }
//! ```

pub mod openai;
