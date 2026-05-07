use std::collections::HashMap;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use async_trait::async_trait;
use crate::error::CheckpointError;
use super::base::*;

/// In-memory cache implementation
pub struct InMemoryCache {
    /// namespace -> key -> (type_tag, bytes, expire_at_unix_secs)
    cache: RwLock<HashMap<CacheNamespace, HashMap<String, (String, Vec<u8>, Option<f64>)>>>,
}

impl InMemoryCache {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseCache for InMemoryCache {
    fn get(&self, keys: &[(CacheNamespace, String)]) -> Result<HashMap<FullKey, JsonValue>, CheckpointError> {
        let cache = self.cache.read();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let mut result = HashMap::new();
        for (namespace, key) in keys {
            if let Some(ns_cache) = cache.get(namespace) {
                if let Some((_tag, bytes, expire_at)) = ns_cache.get(key) {
                    if let Some(exp) = expire_at {
                        if now > *exp {
                            continue; // expired
                        }
                    }
                    if let Ok(val) = serde_json::from_slice(bytes) {
                        result.insert((namespace.clone(), key.clone()), val);
                    }
                }
            }
        }
        Ok(result)
    }

    fn set(&self, pairs: &[(FullKey, JsonValue, Option<i64>)]) -> Result<(), CheckpointError> {
        let mut cache = self.cache.write();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        for ((namespace, key), value, ttl_secs) in pairs {
            let bytes = serde_json::to_vec(value)
                .map_err(|e| CheckpointError::Storage(e.to_string()))?;
            let expire_at = ttl_secs.map(|ttl| now + ttl as f64);
            cache.entry(namespace.clone())
                .or_default()
                .insert(key.clone(), ("json".to_string(), bytes, expire_at));
        }
        Ok(())
    }

    fn clear(&self, namespaces: Option<&[CacheNamespace]>) -> Result<(), CheckpointError> {
        let mut cache = self.cache.write();
        match namespaces {
            Some(ns_list) => {
                for ns in ns_list {
                    cache.remove(ns);
                }
            }
            None => {
                cache.clear();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cache() {
        let cache = InMemoryCache::new();
        let ns = vec!["test".to_string()];
        let key = "k1".to_string();

        // Set
        cache.set(&[((ns.clone(), key.clone()), serde_json::json!("hello"), None)]).unwrap();

        // Get
        let result = cache.get(&[(ns.clone(), key.clone())]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[&(ns, key)], serde_json::json!("hello"));
    }

    #[test]
    fn test_cache_miss() {
        let cache = InMemoryCache::new();
        let result = cache.get(&[(vec!["ns".to_string()], "missing".to_string())]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cache_clear() {
        let cache = InMemoryCache::new();
        let ns = vec!["test".to_string()];
        cache.set(&[((ns.clone(), "k1".to_string()), serde_json::json!(1), None)]).unwrap();
        cache.clear(Some(&[ns.clone()])).unwrap();
        let result = cache.get(&[(ns, "k1".to_string())]).unwrap();
        assert!(result.is_empty());
    }
}
