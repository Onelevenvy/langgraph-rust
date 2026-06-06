use serde_json::Value as JsonValue;
use langgraph_checkpoint::error::ChannelError;

/// The erased channel trait. All channel types implement this.
///
/// Values flow through as `serde_json::Value` for checkpoint compatibility.
/// This is the critical type-erasure strategy that allows the Pregel engine
/// to work with heterogeneous channel types.
pub trait Channel: Send + Sync + 'static {
    /// Return a serializable checkpoint of this channel's state.
    /// Returns None if the channel is empty (MISSING).
    fn checkpoint(&self) -> Option<JsonValue>;

    /// Restore channel state from a checkpoint.
    fn from_checkpoint(&self, checkpoint: Option<&JsonValue>) -> Box<dyn Channel>;

    /// Apply a batch of updates. Returns true if the channel was modified.
    fn update(&self, values: &[JsonValue]) -> Result<bool, ChannelError>;

    /// Get the current value. Returns Err(EmptyChannel) if empty.
    fn get(&self) -> Result<JsonValue, ChannelError>;

    /// Notify that a subscribed task consumed the value.
    /// Returns true if the channel was modified.
    fn consume(&self) -> bool {
        false
    }

    /// Notify that the Pregel run is finishing.
    /// Returns true if the channel was modified.
    fn finish(&self) -> bool {
        false
    }

    /// Return true if the channel has a value available.
    fn is_available(&self) -> bool;

    /// Clone this channel (for checkpoint restoration).
    fn clone_channel(&self) -> Box<dyn Channel>;

    /// Return the name/key for this channel.
    fn name(&self) -> &str;
}
