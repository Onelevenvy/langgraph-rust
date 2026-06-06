use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// Reducer function type: (current, update) -> new
pub type ReducerFn = fn(&JsonValue, &JsonValue) -> JsonValue;

/// Applies a binary operator to accumulate values.
///
/// Created when a state key uses a reducer (e.g., Annotated[list, add_messages]).
/// Supports Overwrite to bypass the reducer.
pub struct BinaryOperatorAggregate {
    key: String,
    value: RwLock<Option<JsonValue>>,
    reducer: ReducerFn,
}

impl BinaryOperatorAggregate {
    pub fn new(key: impl Into<String>, reducer: ReducerFn) -> Self {
        Self {
            key: key.into(),
            value: RwLock::new(None),
            reducer,
        }
    }
}

impl Channel for BinaryOperatorAggregate {
    fn checkpoint(&self) -> Option<JsonValue> {
        self.value.read().clone()
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(checkpoint.cloned()),
            reducer: self.reducer,
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }

        let mut guard = self.value.write();
        let mut seen_overwrite = false;

        for val in values {
            // Check for Overwrite pattern: {"__overwrite__": value}
            if let Some(obj) = val.as_object() {
                if let Some(overwrite_val) = obj.get("__overwrite__") {
                    if seen_overwrite {
                        return Err(ChannelError::InvalidUpdate(
                            "Received multiple Overwrite values in a single update".to_string(),
                        ));
                    }
                    *guard = Some(overwrite_val.clone());
                    seen_overwrite = true;
                    continue;
                }
            }

            // If we've seen an Overwrite, skip non-Overwrite values
            if seen_overwrite {
                continue;
            }

            match guard.as_ref() {
                Some(current) => {
                    let new_val = (self.reducer)(current, val);
                    *guard = Some(new_val);
                }
                None => {
                    *guard = Some(val.clone());
                }
            }
        }
        Ok(true)
    }

    fn get(&self) -> Result<JsonValue, ChannelError> {
        self.value
            .read()
            .clone()
            .ok_or(ChannelError::EmptyChannel)
    }

    fn is_available(&self) -> bool {
        self.value.read().is_some()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(self.value.read().clone()),
            reducer: self.reducer,
        })
    }

    fn name(&self) -> &str {
        &self.key
    }
}

/// Common reducer: append arrays
pub fn append_reducer(current: &JsonValue, update: &JsonValue) -> JsonValue {
    let mut result = match current {
        JsonValue::Array(arr) => arr.clone(),
        other => vec![other.clone()],
    };
    match update {
        JsonValue::Array(arr) => result.extend(arr.iter().cloned()),
        other => result.push(other.clone()),
    }
    JsonValue::Array(result)
}

/// Common reducer: merge objects
pub fn merge_reducer(current: &JsonValue, update: &JsonValue) -> JsonValue {
    match (current, update) {
        (JsonValue::Object(curr), JsonValue::Object(upd)) => {
            let mut merged = curr.clone();
            for (k, v) in upd {
                merged.insert(k.clone(), v.clone());
            }
            JsonValue::Object(merged)
        }
        _ => update.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_reducer() {
        let ch = BinaryOperatorAggregate::new("items", append_reducer);
        ch.update(&[serde_json::json!([1, 2])]).unwrap();
        ch.update(&[serde_json::json!([3, 4])]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!([1, 2, 3, 4]));
    }

    #[test]
    fn test_merge_reducer() {
        let ch = BinaryOperatorAggregate::new("state", merge_reducer);
        ch.update(&[serde_json::json!({"a": 1})]).unwrap();
        ch.update(&[serde_json::json!({"b": 2})]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_overwrite() {
        let ch = BinaryOperatorAggregate::new("items", append_reducer);
        ch.update(&[serde_json::json!([1, 2])]).unwrap();
        ch.update(&[serde_json::json!({"__overwrite__": [99]})]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!([99]));
    }

    #[test]
    fn test_checkpoint_restore() {
        let ch = BinaryOperatorAggregate::new("items", append_reducer);
        ch.update(&[serde_json::json!([1, 2])]).unwrap();

        let cp = ch.checkpoint();
        let restored = ch.from_checkpoint(cp.as_ref());
        assert_eq!(restored.get().unwrap(), serde_json::json!([1, 2]));

        restored.update(&[serde_json::json!([3])]).unwrap();
        assert_eq!(restored.get().unwrap(), serde_json::json!([1, 2, 3]));
    }
}
