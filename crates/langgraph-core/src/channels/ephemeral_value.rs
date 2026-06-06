use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// Stores a value for exactly one step, then clears.
///
/// Has a `guard` flag (default true) that rejects multiple writes per step.
/// Used for the START input channel and branch channels.
pub struct EphemeralValue {
    key: String,
    value: RwLock<Option<JsonValue>>,
    guard: bool,
}

impl EphemeralValue {
    pub fn new(key: impl Into<String>, guard: bool) -> Self {
        Self {
            key: key.into(),
            value: RwLock::new(None),
            guard,
        }
    }
}

impl Channel for EphemeralValue {
    fn checkpoint(&self) -> Option<JsonValue> {
        // Match Python: persist the current value in checkpoint
        self.value.read().clone()
    }

    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(checkpoint.cloned()),
            guard: self.guard,
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            // Match Python: empty update clears the value if it exists
            let mut guard = self.value.write();
            let changed = guard.is_some();
            *guard = None;
            return Ok(changed);
        }
        if self.guard && values.len() > 1 {
            return Err(ChannelError::InvalidUpdate(format!(
                "EphemeralValue channel '{}' received {} values with guard enabled",
                self.key,
                values.len()
            )));
        }
        let new_val = values.last().unwrap().clone();
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

    fn consume(&self) -> bool {
        let changed = self.value.read().is_some();
        *self.value.write() = None;
        changed
    }

    fn is_available(&self) -> bool {
        self.value.read().is_some()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(self.value.read().clone()),
            guard: self.guard,
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
    fn test_ephemeral_basic() {
        let ch = EphemeralValue::new("branch:to:agent", true);
        ch.update(&[serde_json::json!("go")]).unwrap();
        assert!(ch.is_available());
        assert_eq!(ch.get().unwrap(), serde_json::json!("go"));
    }

    #[test]
    fn test_ephemeral_guard() {
        let ch = EphemeralValue::new("branch", true);
        let result = ch.update(&[serde_json::json!(1), serde_json::json!(2)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ephemeral_no_guard() {
        let ch = EphemeralValue::new("branch", false);
        ch.update(&[serde_json::json!(1), serde_json::json!(2)]).unwrap();
        assert_eq!(ch.get().unwrap(), serde_json::json!(2));
    }

    #[test]
    fn test_ephemeral_consume() {
        let ch = EphemeralValue::new("branch", true);
        ch.update(&[serde_json::json!("go")]).unwrap();
        ch.consume();
        assert!(!ch.is_available());
    }

    #[test]
    fn test_ephemeral_checkpointed() {
        let ch = EphemeralValue::new("branch", true);
        ch.update(&[serde_json::json!("go")]).unwrap();
        // Match Python: checkpoint preserves the value
        assert_eq!(ch.checkpoint(), Some(serde_json::json!("go")));
    }
}
