use std::collections::HashMap;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use async_trait::async_trait;
use chrono::Utc;
use crate::error::StoreError;
use super::base::*;

/// In-memory store implementation.
pub struct InMemoryStore {
    data: RwLock<HashMap<Vec<String>, HashMap<String, Item>>>,
    index_config: Option<IndexConfig>,
}

impl InMemoryStore {
    pub fn new(index: Option<IndexConfig>) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            index_config: index,
        }
    }

    fn cosine_similarity(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let dot: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let norm_x: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        let norm_y: f64 = y.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm_x == 0.0 || norm_y == 0.0 {
            0.0
        } else {
            dot / (norm_x * norm_y)
        }
    }

    fn does_match(condition: &MatchCondition, namespace: &[String]) -> bool {
        match condition.match_type {
            NamespaceMatchType::Prefix => {
                if condition.path.len() > namespace.len() {
                    return false;
                }
                condition.path.iter().zip(namespace.iter()).all(|(seg, ns)| {
                    match seg {
                        NamespacePathSegment::Literal(s) => s == ns,
                        NamespacePathSegment::Wildcard => true,
                    }
                })
            }
            NamespaceMatchType::Suffix => {
                if condition.path.len() > namespace.len() {
                    return false;
                }
                let offset = namespace.len() - condition.path.len();
                condition.path.iter().zip(namespace[offset..].iter()).all(|(seg, ns)| {
                    match seg {
                        NamespacePathSegment::Literal(s) => s == ns,
                        NamespacePathSegment::Wildcard => true,
                    }
                })
            }
        }
    }

    fn compare_values(item_value: &JsonValue, filter_value: &JsonValue) -> bool {
        match filter_value {
            JsonValue::Object(obj) => {
                for (op, expected) in obj {
                    match op.as_str() {
                        "$eq" => { if item_value != expected { return false; } }
                        "$ne" => { if item_value == expected { return false; } }
                        "$gt" => {
                            if let (Some(a), Some(b)) = (item_value.as_f64(), expected.as_f64()) {
                                if a <= b { return false; }
                            }
                        }
                        "$gte" => {
                            if let (Some(a), Some(b)) = (item_value.as_f64(), expected.as_f64()) {
                                if a < b { return false; }
                            }
                        }
                        "$lt" => {
                            if let (Some(a), Some(b)) = (item_value.as_f64(), expected.as_f64()) {
                                if a >= b { return false; }
                            }
                        }
                        "$lte" => {
                            if let (Some(a), Some(b)) = (item_value.as_f64(), expected.as_f64()) {
                                if a > b { return false; }
                            }
                        }
                        _ => {}
                    }
                }
                true
            }
            _ => item_value == filter_value,
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new(None)
    }
}

#[async_trait]
impl BaseStore for InMemoryStore {
    fn batch(&self, ops: &[Op]) -> Result<Vec<StoreResult>, StoreError> {
        let mut results = Vec::with_capacity(ops.len());

        for op in ops {
            match op {
                Op::Get(get_op) => {
                    let data = self.data.read();
                    let item = data.get(&get_op.namespace)
                        .and_then(|ns| ns.get(&get_op.key))
                        .cloned();
                    results.push(StoreResult::Item(item));
                }
                Op::Search(search_op) => {
                    let data = self.data.read();
                    let mut items: Vec<SearchItem> = Vec::new();

                    for (ns, ns_data) in data.iter() {
                        if search_op.namespace_prefix.len() > ns.len() {
                            continue;
                        }
                        let prefix_match = search_op.namespace_prefix.iter()
                            .zip(ns.iter())
                            .all(|(a, b)| a == b);
                        if !prefix_match {
                            continue;
                        }

                        for item in ns_data.values() {
                            if let Some(ref filter) = search_op.filter {
                                let mut matches = true;
                                for (k, v) in filter {
                                    match item.value.get(k.as_str()) {
                                        Some(field_val) => {
                                            if !Self::compare_values(field_val, v) {
                                                matches = false;
                                                break;
                                            }
                                        }
                                        None => { matches = false; break; }
                                    }
                                }
                                if !matches {
                                    continue;
                                }
                            }

                            items.push(SearchItem {
                                item: item.clone(),
                                score: None,
                            });
                        }
                    }

                    items.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                    let total = items.len();
                    let start = search_op.offset.min(total);
                    let end = (start + search_op.limit).min(total);
                    items = items[start..end].to_vec();

                    results.push(StoreResult::SearchItems(items));
                }
                Op::Put(put_op) => {
                    let mut data = self.data.write();
                    if let Some(ref value) = put_op.value {
                        let now = Utc::now();
                        let item = Item {
                            value: value.clone(),
                            key: put_op.key.clone(),
                            namespace: put_op.namespace.clone(),
                            created_at: data.get(&put_op.namespace)
                                .and_then(|ns| ns.get(&put_op.key))
                                .map(|item| item.created_at)
                                .unwrap_or(now),
                            updated_at: now,
                        };
                        data.entry(put_op.namespace.clone())
                            .or_default()
                            .insert(put_op.key.clone(), item);
                    } else {
                        if let Some(ns) = data.get_mut(&put_op.namespace) {
                            ns.remove(&put_op.key);
                        }
                    }
                    results.push(StoreResult::None);
                }
                Op::ListNamespaces(list_op) => {
                    let data = self.data.read();
                    let mut namespaces: Vec<Vec<String>> = data.keys().cloned().collect();

                    if let Some(ref conditions) = list_op.match_conditions {
                        namespaces.retain(|ns| {
                            conditions.iter().all(|cond| Self::does_match(cond, ns))
                        });
                    }

                    if let Some(depth) = list_op.max_depth {
                        for ns in &mut namespaces {
                            ns.truncate(depth);
                        }
                        namespaces.sort();
                        namespaces.dedup();
                    }

                    let total = namespaces.len();
                    let start = list_op.offset.min(total);
                    let end = (start + list_op.limit).min(total);
                    namespaces = namespaces[start..end].to_vec();

                    results.push(StoreResult::Namespaces(namespaces));
                }
            }
        }

        Ok(results)
    }

    async fn abatch(&self, ops: &[Op]) -> Result<Vec<StoreResult>, StoreError> {
        self.batch(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_put_and_get() {
        let store = InMemoryStore::new(None);
        store.put(&["users", "1"], "profile", json!({"name": "Alice"}), PutIndex::Default, None).unwrap();

        let item = store.get(&["users", "1"], "profile", None).unwrap();
        assert!(item.is_some());
        assert_eq!(item.unwrap().value, json!({"name": "Alice"}));
    }

    #[test]
    fn test_delete() {
        let store = InMemoryStore::new(None);
        store.put(&["users", "1"], "profile", json!({"name": "Alice"}), PutIndex::Default, None).unwrap();
        store.delete(&["users", "1"], "profile").unwrap();

        let item = store.get(&["users", "1"], "profile", None).unwrap();
        assert!(item.is_none());
    }

    #[test]
    fn test_search_with_filter() {
        let store = InMemoryStore::new(None);
        store.put(&["users"], "alice", json!({"age": 30, "name": "Alice"}), PutIndex::Default, None).unwrap();
        store.put(&["users"], "bob", json!({"age": 25, "name": "Bob"}), PutIndex::Default, None).unwrap();

        let mut filter = HashMap::new();
        filter.insert("age".to_string(), json!({"$gte": 28}));

        let results = store.search(&["users"], None, Some(&filter), 10, 0, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "alice");
    }
}
