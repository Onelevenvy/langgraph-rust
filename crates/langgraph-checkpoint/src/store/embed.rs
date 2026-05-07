use async_trait::async_trait;

/// Embeddings trait for vector search in stores.
///
/// Replaces langchain-core's Embeddings base class.
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// Embed a list of documents
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, EmbeddingError>;

    /// Embed a single query
    fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbeddingError>;

    /// Async embed documents (default delegates to sync)
    async fn aembed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embed_documents(&refs)
    }

    /// Async embed query (default delegates to sync)
    async fn aembed_query(&self, text: String) -> Result<Vec<f64>, EmbeddingError> {
        self.embed_query(&text)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("embedding error: {0}")]
    Error(String),
}

/// A simple embeddings implementation wrapping a closure
pub struct EmbeddingsLambda<F>
where
    F: Fn(&[&str]) -> Result<Vec<Vec<f64>>, EmbeddingError> + Send + Sync,
{
    func: F,
}

impl<F> EmbeddingsLambda<F>
where
    F: Fn(&[&str]) -> Result<Vec<Vec<f64>>, EmbeddingError> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

#[async_trait]
impl<F> Embeddings for EmbeddingsLambda<F>
where
    F: Fn(&[&str]) -> Result<Vec<Vec<f64>>, EmbeddingError> + Send + Sync,
{
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f64>>, EmbeddingError> {
        (self.func)(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbeddingError> {
        let results = (self.func)(&[text])?;
        results.into_iter().next().ok_or_else(|| EmbeddingError::Error("no embedding returned".to_string()))
    }
}

/// Extract text from a JSON value at a given path
pub fn get_text_at_path(obj: &serde_json::Value, path: &[&str]) -> Vec<String> {
    let mut current = obj;
    for segment in path {
        current = match current.get(*segment) {
            Some(v) => v,
            None => return vec![],
        };
    }
    match current {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![current.to_string()],
    }
}
