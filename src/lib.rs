//! `langgraph` is a framework for building stateful, multi-actor applications with LLMs.
//!
//! This crate is the main entry point (umbrella crate) that re-exports the core engine
//! and optional ecosystem crates like checkpointers, prebuilt agents, tracing, and model providers.

// Re-export core engine by default
pub use langgraph_core_rs::*;

// Re-export core checkpointing traits by default
pub mod checkpoint {
    pub use langgraph_checkpoint::*;
}

// Re-export derive macros by default
pub use langgraph_derive::*;

// Optional re-exports controlled by cargo features
#[cfg(feature = "prebuilt")]
pub mod prebuilt {
    pub use langgraph_prebuilt::*;
}

#[cfg(feature = "sqlite")]
pub mod sqlite {
    pub use langgraph_checkpoint_sqlite::*;
}

#[cfg(feature = "postgres")]
pub mod postgres {
    pub use langgraph_checkpoint_postgres::*;
}

#[cfg(feature = "providers")]
pub mod providers {
    pub use langgraph_providers::*;
}

#[cfg(feature = "tracing")]
pub mod tracing {
    pub use langgraph_tracing::*;
}
