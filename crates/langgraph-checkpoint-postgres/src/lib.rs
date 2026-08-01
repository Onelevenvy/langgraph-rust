//! Postgres checkpoint saver implementation using sqlx.

pub mod queries;
pub mod saver;

pub use saver::PostgresSaver;
