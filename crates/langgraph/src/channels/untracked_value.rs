use parking_lot::RwLock;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;
use super::base::Channel;

/// A channel whose writes are NOT persisted in checkpoints.
/// Used for volatile data that should not survive a restart.
pub struct UntrackedValue {
    key: String,
    value: RwLock<Option<JsonValue>>,
}

impl UntrackedValue {
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: RwLock::new(None),
        }
    }
}

impl Channel for UntrackedValue {
    fn checkpoint(&self) -> Option<JsonValue> {
        // Never persist
        None
    }

    fn from_checkpoint(&self, _checkpoint: Option<&JsonValue>) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(None),
        })
    }

    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError> {
        if values.is_empty() {
            return Ok(false);
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

    fn is_available(&self) -> bool {
        self.value.read().is_some()
    }

    fn clone_channel(&self) -> Box<dyn Channel> {
        Box::new(Self {
            key: self.key.clone(),
            value: RwLock::new(None), // Don't clone value for untracked
        })
    }

    fn name(&self) -> &str {
        &self.key
    }
}
