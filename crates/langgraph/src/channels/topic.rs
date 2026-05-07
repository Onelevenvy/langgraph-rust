use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// PubSub topic channel.
///
/// Can accumulate values (`accumulate=true`) or clear each step.
/// Used for the TASKS channel (holds Send objects).
pub struct Topic {
    key: String,
    values: RwLock<Vec<JsonValue>>,
    accumulate: bool,
}

impl Topic {
    pub fn new(key: impl Into<String>, accumulate: bool) -> Self {
        Self {
            key: key.into(),
            values: RwLock::new(Vec::new()),
            accumulate,
        }
    }
}

impl Channel for Topic {
    fn checkpoint(&self) -> Option<JsonValue> {
        let vals = self.values.read();
        if vals.is_empty() {
            None
        } else {
            Some(JsonValue::Array(vals.clone()))
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        let values = match checkpoint {
            Some(JsonValue::Array(arr)) => arr.clone(),
            Some(other) => vec![other.clone()],
            None => Vec::new(),
        };
        Box::new(Self {
            key: self.key.clone(),
            values: RwLock::new(values),
            accumulate: self.accumulate,
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }
        let mut guard = self.values.write();
        for val in values {
            match val {
                JsonValue::Array(arr) => guard.extend(arr.iter().cloned()),
                other => guard.push(other.clone()),
            }
        }
        Ok(true)
    }

    fn get(&self) -> Result<JsonValue, ChannelError> {
        let vals = self.values.read();
        if vals.is_empty() {
            Err(ChannelError::EmptyChannel)
        } else {
            Ok(JsonValue::Array(vals.clone()))
        }
    }

    fn consume(&self) -> bool {
        if !self.accumulate {
            let changed = !self.values.read().is_empty();
            self.values.write().clear();
            changed
        } else {
            false
        }
    }

    fn is_available(&self) -> bool {
        !self.values.read().is_empty()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            values: RwLock::new(self.values.read().clone()),
            accumulate: self.accumulate,
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
    fn test_topic_accumulate() {
        let ch = Topic::new("tasks", true);
        ch.update(&[serde_json::json!("a")]).unwrap();
        ch.update(&[serde_json::json!("b")]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!(["a", "b"]));
        // consume doesn't clear when accumulate=true
        ch.consume();
        assert!(ch.is_available());
    }

    #[test]
    fn test_topic_no_accumulate() {
        let ch = Topic::new("tasks", false);
        ch.update(&[serde_json::json!("a")]).unwrap();
        ch.update(&[serde_json::json!("b")]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!(["a", "b"]));
        // consume clears when accumulate=false
        ch.consume();
        assert!(!ch.is_available());
    }

    #[test]
    fn test_topic_array_update() {
        let ch = Topic::new("tasks", true);
        ch.update(&[serde_json::json!(["a", "b"]), serde_json::json!(["c"])]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!(["a", "b", "c"]));
    }
}
