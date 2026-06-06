use std::collections::HashSet;
use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// Waits until all named values from a set have been received.
///
/// Used for fan-in edges (e.g., `add_edge(["A", "B"], "C")`).
/// `consume()` resets the seen set.
pub struct NamedBarrierValue {
    key: String,
    names: HashSet<String>,
    seen: RwLock<HashSet<String>>,
    value: RwLock<Option<JsonValue>>,
}

impl NamedBarrierValue {
    pub fn new(key: impl Into<String>, names: HashSet<String>) -> Self {
        Self {
            key: key.into(),
            names,
            seen: RwLock::new(HashSet::new()),
            value: RwLock::new(None),
        }
    }
}

impl Channel for NamedBarrierValue {
    fn checkpoint(&self) -> Option<JsonValue> {
        // Only checkpoint when all names have been seen
        let seen = self.seen.read();
        if seen.len() >= self.names.len() {
            Some(JsonValue::Array(
                seen.iter().map(|s: &String| JsonValue::String(s.clone())).collect(),
            ))
        } else {
            None
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        let seen = match checkpoint {
            Some(JsonValue::Array(arr)) => arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => HashSet::new(),
        };
        Box::new(Self {
            key: self.key.clone(),
            names: self.names.clone(),
            seen: RwLock::new(seen),
            value: RwLock::new(None),
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }
        let mut seen = self.seen.write();
        for val in values {
            if let Some(name) = val.as_str() {
                if self.names.contains(name) {
                    seen.insert(name.to_string());
                }
            }
        }
        // Check if all names are seen
        if seen.len() >= self.names.len() {
            *self.value.write() = Some(JsonValue::Array(
                seen.iter().map(|s: &String| JsonValue::String(s.clone())).collect(),
            ));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn get(&self) -> Result<JsonValue, ChannelError> {
        self.value
            .read()
            .clone()
            .ok_or(ChannelError::EmptyChannel)
    }

    fn consume(&self) -> bool {
        let changed = !self.seen.read().is_empty();
        self.seen.write().clear();
        *self.value.write() = None;
        changed
    }

    fn is_available(&self) -> bool {
        self.value.read().is_some()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            names: self.names.clone(),
            seen: RwLock::new(self.seen.read().clone()),
            value: RwLock::new(self.value.read().clone()),
        })
    }

    fn name(&self) -> &str {
        &self.key
    }
}

/// Like NamedBarrierValue but only available after finish() is called.
pub struct NamedBarrierValueAfterFinish {
    key: String,
    names: HashSet<String>,
    seen: RwLock<HashSet<String>>,
    value: RwLock<Option<JsonValue>>,
    finished: RwLock<bool>,
}

impl NamedBarrierValueAfterFinish {
    pub fn new(key: impl Into<String>, names: HashSet<String>) -> Self {
        Self {
            key: key.into(),
            names,
            seen: RwLock::new(HashSet::new()),
            value: RwLock::new(None),
            finished: RwLock::new(false),
        }
    }
}

impl Channel for NamedBarrierValueAfterFinish {
    fn checkpoint(&self) -> Option<JsonValue> {
        if *self.finished.read() {
            self.value.read().clone()
        } else {
            None
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            names: self.names.clone(),
            seen: RwLock::new(HashSet::new()),
            value: RwLock::new(checkpoint.cloned()),
            finished: RwLock::new(checkpoint.is_some()),
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }
        let mut seen = self.seen.write();
        for val in values {
            if let Some(name) = val.as_str() {
                if self.names.contains(name) {
                    seen.insert(name.to_string());
                }
            }
        }
        Ok(seen.len() >= self.names.len())
    }

    fn get(&self) -> Result<JsonValue, ChannelError> {
        self.value
            .read()
            .clone()
            .ok_or(ChannelError::EmptyChannel)
    }

    fn consume(&self) -> bool {
        self.seen.write().clear();
        *self.value.write() = None;
        *self.finished.write() = false;
        true
    }

    fn finish(&self) -> bool {
        let seen = self.seen.read();
        if seen.len() >= self.names.len() {
            *self.value.write() = Some(JsonValue::Array(
                seen.iter().map(|s: &String| JsonValue::String(s.clone())).collect(),
            ));
            *self.finished.write() = true;
            true
        } else {
            false
        }
    }

    fn is_available(&self) -> bool {
        self.value.read().is_some()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            names: self.names.clone(),
            seen: RwLock::new(self.seen.read().clone()),
            value: RwLock::new(self.value.read().clone()),
            finished: RwLock::new(*self.finished.read()),
        })
    }

    fn name(&self) -> &str {
        &self.key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_waits_for_all() {
        let mut names = HashSet::new();
        names.insert("A".to_string());
        names.insert("B".to_string());

        let ch = NamedBarrierValue::new("join", names);
        ch.update(&[serde_json::json!("A")]).unwrap();
        assert!(!ch.is_available());

        ch.update(&[serde_json::json!("B")]).unwrap();
        assert!(ch.is_available());
    }

    #[test]
    fn test_barrier_consume_resets() {
        let mut names = HashSet::new();
        names.insert("A".to_string());
        names.insert("B".to_string());

        let ch = NamedBarrierValue::new("join", names);
        ch.update(&[serde_json::json!("A")]).unwrap();
        ch.update(&[serde_json::json!("B")]).unwrap();
        assert!(ch.is_available());

        ch.consume();
        assert!(!ch.is_available());
    }
}
