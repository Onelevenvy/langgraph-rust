//! Version-keyed cache of deserialized channel values.
//!
//! Savers that store channel values as version-addressed blobs re-parse
//! every blob on every read. For large channels that rarely change (e.g. a
//! static 128KB context), that re-parse is pure overhead: the same
//! `(thread, ns, channel, version)` tuple maps to the same content for the
//! lifetime of the thread. This cache makes repeated reads of unchanged
//! channels O(clone) instead of O(parse).
//!
//! Correctness relies on version strings being monotonic and unique within a
//! `(thread_id, checkpoint_ns)` pair: the engine assigns `{:032}` counters
//! that only grow, and `update_state` derives `max + 1`. The one place a
//! version can be reused is `delete_thread` + recreation of a thread with the
//! same id, so callers MUST call [`BlobCache::remove_thread`] there.

use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Default maximum number of cached values. On overflow the whole cache is
/// cleared (simple, correct; the cost is a periodic re-parse).
pub const DEFAULT_MAX_ENTRIES: usize = 1024;

/// Key: `(thread_id, checkpoint_ns, channel, version_str)`.
pub type BlobCacheKey = (String, String, String, String);

#[derive(Default)]
pub struct BlobCache {
    inner: RwLock<HashMap<BlobCacheKey, JsonValue>>,
    max_entries: usize,
}

impl BlobCache {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_ENTRIES)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            max_entries,
        }
    }

    /// Return a clone of the cached value, if present.
    pub fn get(&self, key: &BlobCacheKey) -> Option<JsonValue> {
        self.inner.read().get(key).cloned()
    }

    /// Cache a deserialized value, evicting everything when over capacity.
    pub fn insert(&self, key: BlobCacheKey, value: JsonValue) {
        let mut inner = self.inner.write();
        if inner.len() >= self.max_entries {
            inner.clear();
        }
        inner.insert(key, value);
    }

    /// Drop all cached values for a thread. REQUIRED after `delete_thread`:
    /// version strings restart at 1 for a recreated thread with the same id,
    /// and stale entries would otherwise be served as the new values.
    pub fn remove_thread(&self, thread_id: &str) {
        self.inner
            .write()
            .retain(|(tid, _, _, _), _| tid != thread_id);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }
}
