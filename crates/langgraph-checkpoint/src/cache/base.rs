use std::collections::HashMap;
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use crate::error::CheckpointError;

/// Cache namespace: tuple of namespace segments
pub type CacheNamespace = Vec<String>;

/// Full cache key: (namespace, key)
pub type FullKey = (CacheNamespace, String);

/// Base cache trait
#[async_trait]
pub trait BaseCache: Send + Sync {
    /// Get cached values by keys
    fn get(&self, keys: &[(CacheNamespace, String)]) -> Result<HashMap<FullKey, JsonValue>, CheckpointError>;

    /// Set cached values with optional TTL (in seconds)
    fn set(&self, pairs: &[(FullKey, JsonValue, Option<i64>)]) -> Result<(), CheckpointError>;

    /// Clear cache entries, optionally limited to specific namespaces
    fn clear(&self, namespaces: Option<&[CacheNamespace]>) -> Result<(), CheckpointError>;

    // Async mirrors

    async fn aget(&self, keys: Vec<(CacheNamespace, String)>) -> Result<HashMap<FullKey, JsonValue>, CheckpointError> {
        self.get(&keys)
    }

    async fn aset(&self, pairs: Vec<(FullKey, JsonValue, Option<i64>)>) -> Result<(), CheckpointError> {
        self.set(&pairs)
    }

    async fn aclear(&self, namespaces: Option<Vec<CacheNamespace>>) -> Result<(), CheckpointError> {
        let ns_ref = namespaces.as_deref();
        self.clear(ns_ref)
    }
}
