use std::collections::HashMap;
use serde_json::Value as JsonValue;
use crate::channels::Channel;
use crate::constants::{START, RESUME, TASKS};
use crate::types::{Command, CommandGoto};

/// Map user input to channel writes.
///
/// If input is an object, each key maps to a channel.
/// Otherwise, writes the entire value to each input channel.
pub fn map_input(
    input_channels: &[String],
    input: &JsonValue,
) -> Vec<(String, JsonValue)> {
    if let Some(obj) = input.as_object() {
        // Dict input — map matching keys to channels
        input_channels
            .iter()
            .filter_map(|ch| {
                obj.get(ch).map(|v| (ch.clone(), v.clone()))
            })
            .collect()
    } else {
        // Non-object input — write to all channels
        input_channels
            .iter()
            .map(|ch| (ch.clone(), input.clone()))
            .collect()
    }
}

/// Read values from output channels.
///
/// Returns a JSON object with channel names as keys.
pub fn read_channels(
    channels: &HashMap<String, Box<dyn Channel>>,
    output_channels: &[String],
) -> JsonValue {
    let mut map = serde_json::Map::new();
    for ch_name in output_channels {
        if let Some(ch) = channels.get(ch_name) {
            if let Ok(val) = ch.get() {
                map.insert(ch_name.clone(), val);
            }
        }
    }
    JsonValue::Object(map)
}

/// Read a single output channel value.
pub fn read_single_channel(
    channels: &HashMap<String, Box<dyn Channel>>,
    channel_name: &str,
) -> Option<JsonValue> {
    channels.get(channel_name)?.get().ok()
}

/// Map task writes to output updates (for stream_mode="updates").
///
/// Returns `{node_name: {channel: value, ...}}` for each task.
pub fn map_output_updates(
    tasks: &[(String, Vec<(String, JsonValue)>)],
) -> HashMap<String, JsonValue> {
    let mut updates = HashMap::new();
    for (task_name, writes) in tasks {
        let mut map = serde_json::Map::new();
        for (chan, val) in writes {
            map.insert(chan.clone(), val.clone());
        }
        updates.insert(task_name.clone(), JsonValue::Object(map));
    }
    updates
}

/// The null task ID used for Command writes (equivalent to Python's NULL_TASK_ID).
pub const NULL_TASK_ID: &str = "00000000-0000-0000-0000-000000000000";

/// Map a Command input to pending writes.
///
/// Returns a list of `(task_id, channel, value)` tuples, matching Python's `map_command`.
///
/// - `goto`: writes to `branch:to:<name>` channels or TASKS channel for Send
/// - `resume`: writes to RESUME channel
/// - `update`: writes to state channels
pub fn map_command(cmd: &Command) -> Vec<(String, String, JsonValue)> {
    let mut writes = Vec::new();

    // Handle goto
    for goto in &cmd.goto {
        match goto {
            CommandGoto::Node(name) => {
                writes.push((
                    NULL_TASK_ID.to_string(),
                    format!("branch:to:{}", name),
                    JsonValue::String(START.to_string()),
                ));
            }
            CommandGoto::Send(send) => {
                writes.push((
                    NULL_TASK_ID.to_string(),
                    TASKS.to_string(),
                    serde_json::to_value(send).unwrap_or_default(),
                ));
            }
        }
    }

    // Handle resume
    if let Some(ref resume) = cmd.resume {
        writes.push((
            NULL_TASK_ID.to_string(),
            RESUME.to_string(),
            resume.clone(),
        ));
    }

    // Handle update
    if let Some(ref update) = cmd.update {
        if let Some(obj) = update.as_object() {
            for (k, v) in obj {
                writes.push((
                    NULL_TASK_ID.to_string(),
                    k.clone(),
                    v.clone(),
                ));
            }
        }
    }

    writes
}
