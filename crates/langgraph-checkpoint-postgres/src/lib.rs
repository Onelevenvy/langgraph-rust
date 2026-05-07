//! Postgres checkpoint saver implementation using sqlx.

pub mod saver;
pub mod queries;

pub use saver::PostgresSaver;
