use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// Default channel for state keys. Stores exactly one value.
///
/// Raises InvalidUpdateError if multiple values arrive in one step.
pub struct LastValue {
    key: String,
    value: RwLock<Option<JsonValue>>,
}

impl LastValue {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: RwLock::new(None),
        }
    }
}

impl Channel for LastValue {
    fn checkpoint(&self) -> Option<JsonValue> {
        self.value.read().clone()
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(checkpoint.cloned()),
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }
        if values.len() > 1 {
            return Err(ChannelError::InvalidUpdate(format!(
                "LastValue channel '{}' received {} values in one step, expected at most 1",
                self.key,
                values.len()
            )));
        }
        let new_val = values[0].clone();
        let mut guard = self.value.write();
        let changed = guard.as_ref() != Some(&new_val);
        *guard = Some(new_val);
        Ok(changed)
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
        })
    }

    fn name(&self) -> &str {
        &self.key
    }
}

/// Like LastValue but only available after finish() is called.
/// Clears on consume. Used for `defer=True` nodes.
pub struct LastValueAfterFinish {
    key: String,
    value: RwLock<Option<JsonValue>>,
    /// Value waiting to be published after finish
    pending: RwLock<Option<JsonValue>>,
    /// Whether finish() has been called
    finished: RwLock<bool>,
}

impl LastValueAfterFinish {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: RwLock::new(None),
            pending: RwLock::new(None),
            finished: RwLock::new(false),
        }
    }
}

impl Channel for LastValueAfterFinish {
    fn checkpoint(&self) -> Option<JsonValue> {
        let val = self.value.read().clone();
        let pending = self.pending.read().clone();
        let finished = *self.finished.read();
        // Checkpoint both value and pending state
        if finished {
            val
        } else {
            pending
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(checkpoint.cloned()),
            pending: RwLock::new(None),
            finished: RwLock::new(false),
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
        }
        if values.len() > 1 {
            return Err(ChannelError::InvalidUpdate(format!(
                "LastValueAfterFinish channel '{}' received {} values",
                self.key,
                values.len()
            )));
        }
        *self.pending.write() = Some(values[0].clone());
        Ok(true)
    }

    fn get(&self) -> Result<JsonValue, ChannelError> {
        self.value
            .read()
            .clone()
            .ok_or(ChannelError::EmptyChannel)
    }

    fn consume(&self) -> bool {
        *self.value.write() = None;
        true
    }

    fn finish(&self) -> bool {
        let pending = self.pending.write().take();
        if let Some(val) = pending {
            *self.value.write() = Some(val);
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
            value: RwLock::new(self.value.read().clone()),
            pending: RwLock::new(self.pending.read().clone()),
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
    fn test_last_value_basic() {
        let ch = LastValue::new("test");
        assert!(!ch.is_available());

        ch.update(&[serde_json::json!(42)]).unwrap();
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), serde_json::json!(42));
    }

    #[test]
    fn test_last_value_multiple_error() {
        let ch = LastValue::new("test");
        let result = ch.update(&[serde_json::json!(1), serde_json::json!(2)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_last_value_checkpoint() {
        let ch = LastValue::new("test");
        ch.update(&[serde_json::json!("hello")]).unwrap();

        let cp = ch.checkpoint();
        assert_eq!(cp, Some(serde_json::json!("hello")));

        let restored = ch.from_checkpoint(cp.as_ref());
        assert_eq!(restored.get().unwrap(), serde_json::json!("hello"));
    }

    #[test]
    fn test_last_value_after_finish() {
        let ch = LastValueAfterFinish::new("deferred");

        // Update puts value in pending
        ch.update(&[serde_json::json!("data")]).unwrap();
        // Not available until finish
        assert!(!ch.is_available());

        // Finish publishes the value
        ch.finish();
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), serde_json::json!("data"));

        // Consume clears it
        ch.consume();
        assert!(!ch.is_available());
    }
}
