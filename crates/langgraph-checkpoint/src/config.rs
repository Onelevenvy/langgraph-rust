use std::collections::HashMap;
use serde_json::Value as JsonValue;

/// RunnableConfig - the universal configuration type for all LangGraph operations.
///
/// Replaces langchain-core's RunnableConfig TypedDict.
/// The "configurable" key holds runtime configuration like thread_id, checkpoint_id, etc.
pub type RunnableConfig = HashMap<String, JsonValue>;

/// Extension trait for RunnableConfig convenience methods
pub trait RunnableConfigExt {
    fn new() -> Self;
    fn get_configurable(&self) -> Option<&JsonValue>;
    fn get_thread_id(&self) -> Option<&str>;
    fn get_checkpoint_id(&self) -> Option<&str>;
    fn get_checkpoint_ns(&self) -> &str;
    fn get_run_id(&self) -> Option<&str>;
    fn get_recursion_limit(&self) -> Option<u64>;
    fn with_recursion_limit(self, limit: u64) -> Self;
}

impl RunnableConfigExt for RunnableConfig {
    fn new() -> Self {
        HashMap::new()
    }

    fn get_configurable(&self) -> Option<&JsonValue> {
        self.get("configurable")
    }

    fn get_thread_id(&self) -> Option<&str> {
        self.get("configurable")
            .and_then(|c| c.get("thread_id"))
            .and_then(|v| v.as_str())
    }

    fn get_checkpoint_id(&self) -> Option<&str> {
        self.get("configurable")
            .and_then(|c| c.get("checkpoint_id"))
            .and_then(|v| v.as_str())
    }

    fn get_checkpoint_ns(&self) -> &str {
        self.get("configurable")
            .and_then(|c| c.get("checkpoint_ns"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
    }

    fn get_run_id(&self) -> Option<&str> {
        self.get("configurable")
            .and_then(|c| c.get("run_id"))
            .and_then(|v| v.as_str())
    }

    fn get_recursion_limit(&self) -> Option<u64> {
        self.get("recursion_limit").and_then(|v| v.as_u64())
    }

    fn with_recursion_limit(mut self, limit: u64) -> Self {
        self.insert("recursion_limit".to_string(), serde_json::json!(limit));
        self
    }
}

/// Merge two configs, with override taking precedence
pub fn merge_configs(base: &RunnableConfig, override_config: &RunnableConfig) -> RunnableConfig {
    let mut result = base.clone();
    for (k, v) in override_config {
        if k == "configurable" {
            if let (Some(base_conf), Some(override_conf)) = (base.get("configurable"), v.as_object()) {
                if let Some(mut merged) = base_conf.as_object().cloned() {
                    for (ck, cv) in override_conf {
                        merged.insert(ck.clone(), cv.clone());
                    }
                    result.insert(k.clone(), JsonValue::Object(merged));
                    continue;
                }
            }
        }
        result.insert(k.clone(), v.clone());
    }
    result
}

/// Patch a config with specific fields
pub fn patch_config(
    config: &RunnableConfig,
    tags: Option<&[&str]>,
    metadata: Option<&HashMap<String, JsonValue>>,
    callbacks: Option<&JsonValue>,
) -> RunnableConfig {
    let mut result = config.clone();
    if let Some(tags) = tags {
        result.insert("tags".to_string(), JsonValue::Array(
            tags.iter().map(|t| JsonValue::String(t.to_string())).collect()
        ));
    }
    if let Some(metadata) = metadata {
        result.insert("metadata".to_string(), serde_json::to_value(metadata).unwrap_or_default());
    }
    if let Some(callbacks) = callbacks {
        result.insert("callbacks".to_string(), callbacks.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_configs() {
        let mut base = RunnableConfig::new();
        base.insert("tags".to_string(), serde_json::json!(["tag1"]));
        base.insert("configurable".to_string(), serde_json::json!({
            "thread_id": "t1",
            "key1": "val1",
        }));

        let mut override_config = RunnableConfig::new();
        override_config.insert("configurable".to_string(), serde_json::json!({
            "thread_id": "t2",
            "key2": "val2",
        }));

        let merged = merge_configs(&base, &override_config);
        let conf = merged.get("configurable").unwrap();
        assert_eq!(conf.get("thread_id").unwrap().as_str().unwrap(), "t2");
        assert_eq!(conf.get("key1").unwrap().as_str().unwrap(), "val1");
        assert_eq!(conf.get("key2").unwrap().as_str().unwrap(), "val2");
    }

    #[test]
    fn test_config_ext() {
        let mut config = RunnableConfig::new();
        config.insert("configurable".to_string(), serde_json::json!({
            "thread_id": "t1",
            "checkpoint_id": "cp1",
        }));

        assert_eq!(config.get_thread_id(), Some("t1"));
        assert_eq!(config.get_checkpoint_id(), Some("cp1"));
        assert_eq!(config.get_checkpoint_ns(), "");
    }

    #[test]
    fn test_recursion_limit() {
        let config = RunnableConfig::new().with_recursion_limit(10);
        assert_eq!(config.get_recursion_limit(), Some(10));

        let config = RunnableConfig::new();
        assert_eq!(config.get_recursion_limit(), None);
    }
}
