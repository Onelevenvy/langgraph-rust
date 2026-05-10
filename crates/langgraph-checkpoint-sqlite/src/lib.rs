//! SQLite checkpoint saver implementation using sqlx.
//!
//! Mirrors the architecture of `langgraph-checkpoint-postgres`, adapted for
//! SQLite's syntax and feature set. Uses a three-table schema
//! (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) plus a
//! `checkpoint_migrations` table for schema versioning.

pub mod queries;
pub mod saver;

pub use saver::SqliteSaver;
