use uuid::Uuid;

/// Generate a UUIDv6-like monotonically increasing ID.
///
/// In the Python implementation, this uses a custom UUIDv6 implementation
/// for monotonicity. In Rust, we use UUIDv7 (time-ordered) from the uuid crate
/// which provides similar monotonic ordering guarantees.
pub fn uuid6() -> String {
    // Use UUIDv7 (time-ordered) for monotonic checkpoint IDs.
    // This ensures get_tuple's max_by_key always finds the latest checkpoint.
    Uuid::now_v7().to_string()
}

/// Generate a deterministic task ID from checkpoint context.
pub fn task_id(checkpoint_id: &str, namespace: &str, step: i64, name: &str, triggers: &[String]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    checkpoint_id.hash(&mut hasher);
    namespace.hash(&mut hasher);
    step.hash(&mut hasher);
    name.hash(&mut hasher);
    for t in triggers {
        t.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uuid6_unique() {
        let id1 = uuid6();
        let id2 = uuid6();
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
    }

    #[test]
    fn test_task_id_deterministic() {
        let id1 = task_id("cp1", "ns", 0, "agent", &["ch1".to_string()]);
        let id2 = task_id("cp1", "ns", 0, "agent", &["ch1".to_string()]);
        assert_eq!(id1, id2);

        let id3 = task_id("cp1", "ns", 0, "agent", &["ch2".to_string()]);
        assert_ne!(id1, id3);
    }
}
