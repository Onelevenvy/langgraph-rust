use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use chrono::{DateTime, Utc};
use crate::error::StoreError;

/// Sentinel for "not provided" vs "explicitly None"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotProvided;

pub const NOT_PROVIDED: NotProvided = NotProvided;

/// A stored item
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Item {
    pub value: JsonValue,
    pub key: String,
    pub namespace: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl std::hash::Hash for Item {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.key.hash(state);
    }
}

impl Eq for Item {}

/// A search result item with optional score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchItem {
    #[serde(flatten)]
    pub item: Item,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl std::ops::Deref for SearchItem {
    type Target = Item;
    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

/// Get operation
#[derive(Debug, Clone)]
pub struct GetOp {
    pub namespace: Vec<String>,
    pub key: String,
    pub refresh_ttl: bool,
}

/// Search operation
#[derive(Debug, Clone)]
pub struct SearchOp {
    pub namespace_prefix: Vec<String>,
    pub filter: Option<HashMap<String, JsonValue>>,
    pub limit: usize,
    pub offset: usize,
    pub query: Option<String>,
    pub refresh_ttl: bool,
}

/// Put operation
#[derive(Debug, Clone)]
pub struct PutOp {
    pub namespace: Vec<String>,
    pub key: String,
    pub value: Option<JsonValue>, // None = delete
    pub index: PutIndex,
    pub ttl: Option<f64>,
}

/// Index specification for PutOp
#[derive(Debug, Clone)]
pub enum PutIndex {
    /// Don't index
    False,
    /// Index specific fields
    Fields(Vec<String>),
    /// Default indexing
    Default,
}

/// List namespaces operation
#[derive(Debug, Clone)]
pub struct ListNamespacesOp {
    pub match_conditions: Option<Vec<MatchCondition>>,
    pub max_depth: Option<usize>,
    pub limit: usize,
    pub offset: usize,
}

/// Match condition for namespace filtering
#[derive(Debug, Clone)]
pub struct MatchCondition {
    pub match_type: NamespaceMatchType,
    pub path: Vec<NamespacePathSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceMatchType {
    Prefix,
    Suffix,
}

#[derive(Debug, Clone)]
pub enum NamespacePathSegment {
    Literal(String),
    Wildcard,
}

/// All possible operations
#[derive(Debug, Clone)]
pub enum Op {
    Get(GetOp),
    Search(SearchOp),
    Put(PutOp),
    ListNamespaces(ListNamespacesOp),
}

/// All possible results
#[derive(Debug, Clone)]
pub enum StoreResult {
    Item(Option<Item>),
    SearchItems(Vec<SearchItem>),
    Namespaces(Vec<Vec<String>>),
    None,
}

/// TTL configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TTLConfig {
    #[serde(default)]
    pub refresh_on_read: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_ttl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sweep_interval_minutes: Option<f64>,
}

/// Index configuration for vector search
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dims: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,
}

/// Base store trait with batch/abatch pattern
#[async_trait]
pub trait BaseStore: Send + Sync {
    /// Whether this store supports TTL
    fn supports_ttl(&self) -> bool {
        false
    }

    /// Execute a batch of operations synchronously
    fn batch(&self, ops: &[Op]) -> Result<Vec<StoreResult>, StoreError>;

    /// Execute a batch of operations asynchronously
    async fn abatch(&self, ops: &[Op]) -> Result<Vec<StoreResult>, StoreError>;

    // Convenience methods with default implementations

    fn get(&self, namespace: &[&str], key: &str, refresh_ttl: Option<bool>) -> Result<Option<Item>, StoreError> {
        let ops = vec![Op::Get(GetOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            refresh_ttl: refresh_ttl.unwrap_or(true),
        })];
        match self.batch(&ops)?.into_iter().next() {
            Some(StoreResult::Item(item)) => Ok(item),
            _ => Ok(None),
        }
    }

    fn search(
        &self,
        namespace_prefix: &[&str],
        query: Option<&str>,
        filter: Option<&HashMap<String, JsonValue>>,
        limit: usize,
        offset: usize,
        refresh_ttl: Option<bool>,
    ) -> Result<Vec<SearchItem>, StoreError> {
        let ops = vec![Op::Search(SearchOp {
            namespace_prefix: namespace_prefix.iter().map(|s| s.to_string()).collect(),
            filter: filter.cloned(),
            limit,
            offset,
            query: query.map(|s| s.to_string()),
            refresh_ttl: refresh_ttl.unwrap_or(true),
        })];
        match self.batch(&ops)?.into_iter().next() {
            Some(StoreResult::SearchItems(items)) => Ok(items),
            _ => Ok(vec![]),
        }
    }

    fn put(
        &self,
        namespace: &[&str],
        key: &str,
        value: JsonValue,
        index: PutIndex,
        ttl: Option<f64>,
    ) -> Result<(), StoreError> {
        let ops = vec![Op::Put(PutOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            value: Some(value),
            index,
            ttl,
        })];
        self.batch(&ops)?;
        Ok(())
    }

    fn delete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError> {
        let ops = vec![Op::Put(PutOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            value: None,
            index: PutIndex::Default,
            ttl: None,
        })];
        self.batch(&ops)?;
        Ok(())
    }

    fn list_namespaces(
        &self,
        prefix: Option<&[&str]>,
        suffix: Option<&[&str]>,
        max_depth: Option<usize>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Vec<String>>, StoreError> {
        let mut conditions = Vec::new();
        if let Some(p) = prefix {
            conditions.push(MatchCondition {
                match_type: NamespaceMatchType::Prefix,
                path: p.iter().map(|s| NamespacePathSegment::Literal(s.to_string())).collect(),
            });
        }
        if let Some(s) = suffix {
            conditions.push(MatchCondition {
                match_type: NamespaceMatchType::Suffix,
                path: s.iter().map(|seg| NamespacePathSegment::Literal(seg.to_string())).collect(),
            });
        }
        let ops = vec![Op::ListNamespaces(ListNamespacesOp {
            match_conditions: if conditions.is_empty() { None } else { Some(conditions) },
            max_depth,
            limit,
            offset,
        })];
        match self.batch(&ops)?.into_iter().next() {
            Some(StoreResult::Namespaces(ns)) => Ok(ns),
            _ => Ok(vec![]),
        }
    }

    // Async convenience methods

    async fn aget(&self, namespace: &[&str], key: &str, refresh_ttl: Option<bool>) -> Result<Option<Item>, StoreError> {
        let ops = vec![Op::Get(GetOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            refresh_ttl: refresh_ttl.unwrap_or(true),
        })];
        match self.abatch(&ops).await?.into_iter().next() {
            Some(StoreResult::Item(item)) => Ok(item),
            _ => Ok(None),
        }
    }

    async fn aput(
        &self,
        namespace: &[&str],
        key: &str,
        value: JsonValue,
        index: PutIndex,
        ttl: Option<f64>,
    ) -> Result<(), StoreError> {
        let ops = vec![Op::Put(PutOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            value: Some(value),
            index,
            ttl,
        })];
        self.abatch(&ops).await?;
        Ok(())
    }

    async fn adelete(&self, namespace: &[&str], key: &str) -> Result<(), StoreError> {
        let ops = vec![Op::Put(PutOp {
            namespace: namespace.iter().map(|s| s.to_string()).collect(),
            key: key.to_string(),
            value: None,
            index: PutIndex::Default,
            ttl: None,
        })];
        self.abatch(&ops).await?;
        Ok(())
    }
}

/// Validate a namespace (must not be empty, must not contain empty strings)
pub fn validate_namespace(namespace: &[String]) -> Result<(), StoreError> {
    if namespace.is_empty() {
        return Err(StoreError::InvalidNamespace("namespace cannot be empty".to_string()));
    }
    for segment in namespace {
        if segment.is_empty() {
            return Err(StoreError::InvalidNamespace("namespace segment cannot be empty".to_string()));
        }
    }
    Ok(())
}
