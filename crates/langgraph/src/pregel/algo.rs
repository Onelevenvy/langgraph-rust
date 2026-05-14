use std::collections::{HashMap, HashSet};
use serde_json::Value as JsonValue;
use crate::channels::Channel;
use crate::constants::{TASKS, INTERRUPT, RESUME, NULL_TASK_ID, CONFIG_KEY_SCRATCHPAD, RESERVED};
use crate::types::PregelScratchpad;
use super::{PregelNode, PregelExecutableTask, ChannelVersions, TriggerToNodes};

/// Compare two channel versions. Returns true if `a` > `b`.
/// Versions are compared as their string representations.
fn version_gt(a: &JsonValue, b: &JsonValue) -> bool {
    let a_str = match a {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) => n.to_string(),
        _ => return false,
    };
    let b_str = match b {
        JsonValue::String(s) => s.clone(),
        JsonValue::Number(n) => n.to_string(),
        _ => return false,
    };
    a_str > b_str
}

/// Prepare the next batch of tasks for execution.
///
/// This is the "Plan" phase of the BSP cycle. It checks which nodes
/// have trigger channels with newer versions than what the node last saw.
pub fn prepare_next_tasks(
    nodes: &HashMap<String, PregelNode>,
    channels: &HashMap<String, Box<dyn Channel>>,
    config: &langgraph_checkpoint::config::RunnableConfig,
    step: u64,
    versions_seen: &mut HashMap<String, HashMap<String, JsonValue>>,
    trigger_to_nodes: &TriggerToNodes,
    updated_channels: Option<&HashSet<String>>,
    checkpoint_id: &str,
    pending_writes: &[(String, String, JsonValue)],
    channel_versions: &ChannelVersions,
) -> Vec<PregelExecutableTask> {
    let mut tasks = Vec::new();
    let null_version = JsonValue::String("".to_string());

    // Use tracked channel versions (not channel content) for version comparison
    let current_versions = channel_versions;

    // Determine candidate nodes
    let candidates: Vec<String> = if let Some(updated) = updated_channels {
        // Optimization: only check nodes triggered by updated channels
        let mut candidate_set = HashSet::new();
        for chan in updated {
            if let Some(node_names) = trigger_to_nodes.get(chan) {
                candidate_set.extend(node_names.iter().cloned());
            }
        }
        candidate_set.into_iter().collect()
    } else {
        nodes.keys().cloned().collect()
    };

    // Find global resume value from pending writes (from Command(resume=...))
    let null_resume: Option<&JsonValue> = pending_writes.iter().find_map(|(tid, chan, val)| {
        if tid == NULL_TASK_ID && chan == RESUME {
            Some(val)
        } else {
            None
        }
    });

    for name in candidates {
        let node = match nodes.get(&name) {
            Some(n) => n,
            None => continue,
        };

        // Check if any trigger channel has been updated.
        // Mirrors Python's _triggers(): always check both availability AND version,
        // even during resume.
        let should_trigger = if let Some(seen) = versions_seen.get(&name) {
            node.triggers.iter().any(|chan| {
                let chan_available = channels.get(chan).is_some_and(|c| c.is_available());
                let chan_version = current_versions.get(chan).unwrap_or(&null_version);
                let last_seen = seen.get(chan).unwrap_or(&null_version);
                chan_available && version_gt(chan_version, last_seen)
            })
        } else {
            // Never run before — trigger if any trigger channel is available
            node.triggers.iter().any(|chan| {
                channels.get(chan).is_some_and(|c| c.is_available())
            })
        };

        if !should_trigger {
            continue;
        }

        // Gather input from channels
        let input = gather_input(node, channels);

        // Build deterministic task ID
        let task_id = format!("{}:{:04}:PULL:{}", checkpoint_id, step, name);

        // Find task-specific resume value
        let task_resume: Vec<JsonValue> = pending_writes
            .iter()
            .filter(|(tid, chan, _)| tid == &task_id && chan == RESUME)
            .map(|(_, _, val)| val.clone())
            .collect();

        // Create scratchpad for this task
        let scratchpad = create_scratchpad(
            null_resume,
            &task_resume,
            step,
        );

        // Inject scratchpad into config
        let mut task_config = config.clone();
        let configurable = task_config
            .entry("configurable".to_string())
            .or_insert_with(|| JsonValue::Object(serde_json::Map::new()));
        if let Some(conf_obj) = configurable.as_object_mut() {
            conf_obj.insert(
                CONFIG_KEY_SCRATCHPAD.to_string(),
                serde_json::to_value(&scratchpad).unwrap_or_default(),
            );
        }

        tasks.push(PregelExecutableTask {
            name: name.clone(),
            input,
            proc: node.bound.clone(),
            writes: Vec::new(),
            config: task_config,
            triggers: node.triggers.clone(),
            id: task_id,
        });
    }

    tasks
}

/// Gather input for a node from its input channels.
///
/// Always returns a JSON object mapping channel names to their values,
/// matching Python's behavior where node input is always a state dict.
fn gather_input(
    node: &PregelNode,
    channels: &HashMap<String, Box<dyn Channel>>,
) -> JsonValue {
    let mut map = serde_json::Map::new();
    for ch in &node.channels {
        if let Some(channel) = channels.get(ch) {
            if let Ok(val) = channel.get() {
                map.insert(ch.clone(), val);
            }
        }
    }
    JsonValue::Object(map)
}

/// Create a PregelScratchpad for a task.
///
/// The scratchpad contains resume values from three sources:
/// 1. Global null resume (from Command(resume=...))
/// 2. Task-specific resume values
/// 3. Namespace-mapped resume values (not implemented yet)
fn create_scratchpad(
    null_resume: Option<&JsonValue>,
    task_resume: &[JsonValue],
    step: u64,
) -> PregelScratchpad {
    let mut resume_values = task_resume.to_vec();

    // If there's a global null resume and no task-specific resume, use it
    if resume_values.is_empty() {
        if let Some(null_val) = null_resume {
            resume_values.push(null_val.clone());
        }
    }

    PregelScratchpad {
        step,
        interrupt_counter: 0,
        resume: resume_values,
        is_resuming: null_resume.is_some() || !task_resume.is_empty(),
    }
}

/// Apply writes from completed tasks to channels.
///
/// This is the "Update" phase of the BSP cycle. It:
/// 1. Updates versions_seen for each task's trigger channels
/// 2. Computes a single global next_version from the max of all channel versions
/// 3. Consumes trigger channels (flushes ephemeral values) and bumps their versions
/// 4. Groups writes by channel, applies them, and bumps versions
/// 5. Notifies un-updated channels of the new superstep (bump_step)
/// 6. Calls finish() on all channels if no trigger channels were updated
///
/// Returns the set of updated channel names.
pub fn apply_writes(
    channels: &mut HashMap<String, Box<dyn Channel>>,
    tasks: &[PregelExecutableTask],
    versions_seen: &mut HashMap<String, HashMap<String, JsonValue>>,
    channel_versions: &mut ChannelVersions,
    trigger_to_nodes: &TriggerToNodes,
    get_next_version: impl Fn(Option<&JsonValue>) -> JsonValue,
) -> HashSet<String> {
    let mut updated = HashSet::new();

    // if no task has triggers this is applying writes from the null task only
    // so we don't do anything other than update the channels written to
    let bump_step = tasks.iter().any(|t| !t.triggers.is_empty());

    // 1. Update versions_seen for each task's trigger channels
    for task in tasks {
        let seen = versions_seen.entry(task.name.clone()).or_default();
        for trigger in &task.triggers {
            if let Some(ver) = channel_versions.get(trigger) {
                seen.insert(trigger.clone(), ver.clone());
            }
        }
    }

    // 2. Compute a single global next_version from the max of all channel versions.
    //    This mirrors Python's behavior: all channels updated in the same superstep
    //    share the same version "timestamp".
    let max_version = channel_versions.values().max_by(|a, b| {
        version_gt_partial(a, b)
    }).cloned();
    let next_version = get_next_version(max_version.as_ref());

    // 3. Consume trigger channels (flush ephemeral/topic values).
    //    Filter out RESERVED channels (matching Python behavior).
    //    If consume() returns true (state changed), bump the channel version.
    let trigger_channels: HashSet<String> = tasks
        .iter()
        .flat_map(|t| t.triggers.iter().cloned())
        .collect();

    for chan in &trigger_channels {
        if RESERVED.contains(&chan.as_str()) {
            continue;
        }
        if let Some(ch) = channels.get(chan.as_str()) {
            if ch.consume() {
                channel_versions.insert(chan.clone(), next_version.clone());
            }
        }
    }

    // 4. Group writes by channel
    let mut writes_by_channel: HashMap<String, Vec<JsonValue>> = HashMap::new();
    for task in tasks {
        for (chan, val) in &task.writes {
            // Skip special channels
            if chan == TASKS || chan == INTERRUPT {
                continue;
            }
            writes_by_channel
                .entry(chan.clone())
                .or_default()
                .push(val.clone());
        }
    }

    // 5. Apply writes to channels and bump versions
    for (chan, vals) in &writes_by_channel {
        if let Some(ch) = channels.get(chan.as_str()) {
            if ch.update(vals).unwrap_or(false) {
                channel_versions.insert(chan.clone(), next_version.clone());
                // unavailable channels can't trigger tasks, so don't add them
                if ch.is_available() {
                    updated.insert(chan.clone());
                }
            }
        }
    }

    // 6. Channels that weren't updated in this step are notified of a new step.
    //    This allows ephemeral channels to clear themselves and notify downstream.
    if bump_step {
        for (chan, ch) in channels.iter() {
            if ch.is_available() && !updated.contains(chan) {
                if ch.update(&[]).unwrap_or(false) {
                    channel_versions.insert(chan.clone(), next_version.clone());
                    if ch.is_available() {
                        updated.insert(chan.clone());
                    }
                }
            }
        }
    }

    // 7. If this is (tentatively) the last superstep, notify all channels of finish.
    //    If finish() returns true (state changed), bump the channel version.
    if bump_step && !updated.iter().any(|u| trigger_to_nodes.contains_key(u)) {
        for (chan, ch) in channels.iter() {
            if ch.finish() {
                channel_versions.insert(chan.clone(), next_version.clone());
                if ch.is_available() {
                    updated.insert(chan.clone());
                }
            }
        }
    }

    updated
}

/// Helper for comparing versions in max_by.
/// Returns Ordering::Greater if a > b (string-wise).
fn version_gt_partial(a: &JsonValue, b: &JsonValue) -> std::cmp::Ordering {
    let a_str = match a {
        JsonValue::String(s) => s.as_str(),
        JsonValue::Number(n) => return n.to_string().cmp(&b.to_string()),
        _ => return std::cmp::Ordering::Equal,
    };
    let b_str = match b {
        JsonValue::String(s) => s.as_str(),
        JsonValue::Number(n) => return a_str.cmp(&n.to_string()),
        _ => return std::cmp::Ordering::Equal,
    };
    a_str.cmp(b_str)
}

/// Check if we should interrupt before executing the given nodes.
pub fn should_interrupt(
    interrupt_nodes: &HashSet<String>,
    task_names: &[String],
) -> bool {
    task_names.iter().any(|n| interrupt_nodes.contains(n))
}
