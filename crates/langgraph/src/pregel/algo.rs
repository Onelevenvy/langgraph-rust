use std::collections::{HashMap, HashSet};
use serde_json::Value as JsonValue;
use crate::channels::Channel;
use crate::constants::{TASKS, INTERRUPT, RESUME, NULL_TASK_ID, CONFIG_KEY_SCRATCHPAD};
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

    // When resuming, trigger channels may have been consumed (empty) from the
    // prior step. We still want nodes to re-trigger based on version comparison
    // alone, since the resume value in the scratchpad will make interrupt()
    // return instead of throwing.
    let is_resuming = null_resume.is_some();

    for name in candidates {
        let node = match nodes.get(&name) {
            Some(n) => n,
            None => continue,
        };

        // Check if any trigger channel has been updated
        let should_trigger = if let Some(seen) = versions_seen.get(&name) {
            node.triggers.iter().any(|chan| {
                let chan_available = channels.get(chan).is_some_and(|c| c.is_available());
                let chan_version = current_versions.get(chan).unwrap_or(&null_version);
                let last_seen = seen.get(chan).unwrap_or(&null_version);
                if is_resuming {
                    // On resume, version comparison alone is sufficient.
                    // Trigger channels may be empty (consumed) from the prior step.
                    version_gt(chan_version, last_seen)
                } else {
                    chan_available && version_gt(chan_version, last_seen)
                }
            })
        } else {
            // Never run before
            if is_resuming {
                // On resume, trigger if any trigger channel has a version
                // (the node was interrupted before its versions_seen was recorded)
                node.triggers.iter().any(|chan| {
                    current_versions.contains_key(chan)
                })
            } else {
                // Fresh run — trigger if any trigger channel is available
                node.triggers.iter().any(|chan| {
                    channels.get(chan).is_some_and(|c| c.is_available())
                })
            }
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
/// 1. Groups writes by channel
/// 2. Applies them to channels via `update()`
/// 3. Bumps channel versions for changed channels
/// 4. Consumes trigger channels (flushes ephemeral values)
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

    // 1. Update versions_seen for each task's trigger channels
    for task in tasks {
        let seen = versions_seen.entry(task.name.clone()).or_default();
        for trigger in &task.triggers {
            if let Some(ver) = channel_versions.get(trigger) {
                seen.insert(trigger.clone(), ver.clone());
            }
        }
    }

    // 2. Consume trigger channels (flush ephemeral/topic values)
    let trigger_channels: HashSet<String> = tasks
        .iter()
        .flat_map(|t| t.triggers.iter().cloned())
        .collect();

    for chan in &trigger_channels {
        if let Some(ch) = channels.get(chan) {
            ch.consume();
        }
    }

    // 3. Group writes by channel
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

    // 4. Apply writes to channels
    for (chan, vals) in &writes_by_channel {
        if let Some(ch) = channels.get(chan) {
            if ch.update(vals).unwrap_or(false) {
                // Channel value changed — bump version
                let new_ver = get_next_version(channel_versions.get(chan));
                channel_versions.insert(chan.clone(), new_ver);
                updated.insert(chan.clone());
            }
        }
    }

    // 5. Check if we should finish (no more tasks can trigger)
    let any_trigger_updated = updated.iter().any(|u| trigger_to_nodes.contains_key(u));
    if !any_trigger_updated && !updated.is_empty() {
        // No updated channel can trigger any node — call finish() on all channels
        for ch in channels.values() {
            ch.finish();
        }
    }

    updated
}

/// Check if we should interrupt before executing the given nodes.
pub fn should_interrupt(
    interrupt_nodes: &HashSet<String>,
    task_names: &[String],
) -> bool {
    task_names.iter().any(|n| interrupt_nodes.contains(n))
}
