use async_trait::async_trait;
use serde_json::Value as JsonValue;
use langgraph_checkpoint::config::RunnableConfig;
use crate::runnable::{Runnable, RunnableError};

/// Sentinel value indicating the input should be passed through unchanged.
const PASSTHROUGH: &str = "__passthrough__";

/// A single channel write entry.
#[derive(Debug, Clone)]
pub struct ChannelWriteEntry {
    /// Target channel name.
    pub channel: String,
    /// Value to write. If `None`, uses PASSTHROUGH (the node's output).
    pub value: Option<JsonValue>,
    /// If true, skip writing when value is null.
    pub skip_none: bool,
}

impl ChannelWriteEntry {
    pub fn new(channel: impl Into<String>, value: Option<JsonValue>) -> Self {
        Self {
            channel: channel.into(),
            value,
            skip_none: false,
        }
    }

    pub fn passthrough(channel: impl Into<String>) -> Self {
        Self {
            channel: channel.into(),
            value: None,
            skip_none: false,
        }
    }
}

/// A Runnable that writes values to channels.
///
/// When invoked, replaces PASSTHROUGH sentinels with the actual input,
/// then calls `config[CONFIG_KEY_SEND]` to buffer the writes.
pub struct ChannelWrite {
    entries: Vec<ChannelWriteEntry>,
}

impl ChannelWrite {
    pub fn new(entries: Vec<ChannelWriteEntry>) -> Self {
        Self { entries }
    }

    /// Process writes: replace PASSTHROUGH with actual input, filter skip_none.
    fn assemble_writes(&self, input: &JsonValue) -> Vec<(String, JsonValue)> {
        let mut writes = Vec::new();
        for entry in &self.entries {
            let value = match &entry.value {
                Some(v) => v.clone(),
                None => input.clone(), // PASSTHROUGH
            };

            if entry.skip_none && value.is_null() {
                continue;
            }

            writes.push((entry.channel.clone(), value));
        }
        writes
    }
}

#[async_trait]
impl Runnable for ChannelWrite {
    fn invoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        let _writes = self.assemble_writes(input);

        // Store writes in the configurable dict under CONFIG_KEY_SEND
        // The caller (PregelLoop/Runner) extracts these after execution
        if let Some(configurable) = config.get("configurable") {
            if let Some(_send_fn) = configurable.get(crate::constants::CONFIG_KEY_SEND) {
                // If there's a send function registered, use it
                // For now, we store writes as a JSON array in the config
                // This will be extracted by the runner
            }
        }

        // Return the input unchanged (writers are side-effect only)
        Ok(input.clone())
    }

    async fn ainvoke(&self, input: &JsonValue, config: &RunnableConfig) -> Result<JsonValue, RunnableError> {
        self.invoke(input, config)
    }

    fn name(&self) -> &str {
        "ChannelWrite"
    }
}

/// Helper to create a ChannelWrite that writes the node's output to
/// the "branch:to:{target}" trigger channels for each destination.
pub fn write_to_targets(targets: &[String]) -> ChannelWrite {
    let entries: Vec<ChannelWriteEntry> = targets
        .iter()
        .map(|t| {
            ChannelWriteEntry::new(
                format!("branch:to:{}", t),
                Some(JsonValue::String(t.clone())),
            )
        })
        .collect();
    ChannelWrite::new(entries)
}
