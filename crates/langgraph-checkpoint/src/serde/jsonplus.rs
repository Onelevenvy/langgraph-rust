use std::any::Any;
use serde_json::Value as JsonValue;
use crate::error::SerdeError;
use super::base::SerializerProtocol;

/// Msgpack extension type codes, matching Python's jsonplus.py
pub const EXT_CONSTRUCTOR_SINGLE_ARG: i8 = 0;
pub const EXT_CONSTRUCTOR_POS_ARGS: i8 = 1;
pub const EXT_CONSTRUCTOR_KW_ARGS: i8 = 2;
pub const EXT_METHOD_SINGLE_ARG: i8 = 3;
pub const EXT_PYDANTIC_V1: i8 = 4;
pub const EXT_PYDANTIC_V2: i8 = 5;
pub const EXT_NUMPY_ARRAY: i8 = 6;

/// Type tag constants
pub const TAG_NULL: &str = "null";
pub const TAG_BYTES: &str = "bytes";
pub const TAG_BYTEARRAY: &str = "bytearray";
pub const TAG_JSON: &str = "json";
pub const TAG_MSGPACK: &str = "msgpack";
pub const TAG_PICKLE: &str = "pickle";

/// JsonPlusSerializer - the default serializer for LangGraph checkpoints.
///
/// Uses serde_json for JSON serialization and rmp-serde for msgpack.
/// In the Rust port, we simplify the ext-type dispatch since we don't
/// need Python's dynamic module loading.
pub struct JsonPlusSerializer {
    /// Whether to fall back to pickle (not applicable in Rust, kept for API compat)
    pickle_fallback: bool,
}

impl JsonPlusSerializer {
    pub fn new() -> Self {
        Self {
            pickle_fallback: false,
        }
    }

    pub fn with_pickle_fallback(mut self, enabled: bool) -> Self {
        self.pickle_fallback = enabled;
        self
    }
}

impl Default for JsonPlusSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializerProtocol for JsonPlusSerializer {
    fn dumps_typed(&self, obj: &dyn Any) -> Result<(String, Vec<u8>), SerdeError> {
        // Handle null
        if obj.is::<()>() {
            return Ok((TAG_NULL.to_string(), vec![]));
        }

        // Handle raw bytes
        if let Some(bytes) = obj.downcast_ref::<Vec<u8>>() {
            return Ok((TAG_BYTES.to_string(), bytes.clone()));
        }

        // Handle serde_json::Value (serialize to JSON)
        if let Some(val) = obj.downcast_ref::<JsonValue>() {
            let data = serde_json::to_vec(val)?;
            return Ok((TAG_JSON.to_string(), data));
        }

        // Handle String
        if let Some(s) = obj.downcast_ref::<String>() {
            let val = JsonValue::String(s.clone());
            let data = serde_json::to_vec(&val)?;
            return Ok((TAG_JSON.to_string(), data));
        }

        // Handle &str
        if let Some(s) = obj.downcast_ref::<&str>() {
            let val = JsonValue::String(s.to_string());
            let data = serde_json::to_vec(&val)?;
            return Ok((TAG_JSON.to_string(), data));
        }

        // For types implementing Serialize, try msgpack first, then JSON
        // Since we can't check Serialize trait at runtime with dyn Any,
        // we default to JSON for known types
        Err(SerdeError::NotSerializable(
            format!("Type not directly serializable through Any: {:?}", obj.type_id())
        ))
    }

    fn loads_typed(&self, tag: &str, data: &[u8]) -> Result<Box<dyn Any>, SerdeError> {
        match tag {
            TAG_NULL => Ok(Box::new(())),
            TAG_BYTES => Ok(Box::new(data.to_vec())),
            TAG_BYTEARRAY => Ok(Box::new(data.to_vec())),
            TAG_JSON => {
                let val: JsonValue = serde_json::from_slice(data)?;
                Ok(Box::new(val))
            }
            TAG_MSGPACK => {
                let val: JsonValue = rmp_serde::from_slice(data)
                    .map_err(|e| SerdeError::Msgpack(e.to_string()))?;
                Ok(Box::new(val))
            }
            TAG_PICKLE => {
                Err(SerdeError::NotSerializable(
                    "Pickle deserialization is not supported in Rust".to_string()
                ))
            }
            _ => Err(SerdeError::UnknownTag(tag.to_string())),
        }
    }
}

/// Helper to serialize a serde-compatible value to msgpack bytes
pub fn to_msgpack_bytes<T: serde::Serialize>(val: &T) -> Result<Vec<u8>, SerdeError> {
    rmp_serde::to_vec_named(val).map_err(|e| SerdeError::Msgpack(e.to_string()))
}

/// Helper to deserialize msgpack bytes to a serde-compatible type
pub fn from_msgpack_bytes<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, SerdeError> {
    rmp_serde::from_slice(data).map_err(|e| SerdeError::Msgpack(e.to_string()))
}

/// Helper to serialize a serde-compatible value to JSON bytes
pub fn to_json_bytes<T: serde::Serialize>(val: &T) -> Result<Vec<u8>, SerdeError> {
    serde_json::to_vec(val).map_err(SerdeError::Json)
}

/// Helper to deserialize JSON bytes to a serde-compatible type
pub fn from_json_bytes<T: serde::de::DeserializeOwned>(data: &[u8]) -> Result<T, SerdeError> {
    serde_json::from_slice(data).map_err(SerdeError::Json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_roundtrip() {
        let serde = JsonPlusSerializer::new();
        let (tag, data) = serde.dumps_typed(&()).unwrap();
        assert_eq!(tag, TAG_NULL);
        assert!(data.is_empty());
        let result = serde.loads_typed(&tag, &data).unwrap();
        assert!(result.is::<()>());
    }

    #[test]
    fn test_bytes_roundtrip() {
        let serde = JsonPlusSerializer::new();
        let input: Vec<u8> = vec![1, 2, 3, 4, 5];
        let (tag, data) = serde.dumps_typed(&input).unwrap();
        assert_eq!(tag, TAG_BYTES);
        assert_eq!(data, input);
        let result = serde.loads_typed(&tag, &data).unwrap();
        let output = result.downcast_ref::<Vec<u8>>().unwrap();
        assert_eq!(*output, input);
    }

    #[test]
    fn test_json_value_roundtrip() {
        let serde = JsonPlusSerializer::new();
        let input = serde_json::json!({"key": "value", "num": 42});
        let (tag, data) = serde.dumps_typed(&input).unwrap();
        assert_eq!(tag, TAG_JSON);
        let result = serde.loads_typed(&tag, &data).unwrap();
        let output = result.downcast_ref::<JsonValue>().unwrap();
        assert_eq!(*output, input);
    }

    #[test]
    fn test_string_roundtrip() {
        let serde = JsonPlusSerializer::new();
        let input = String::from("hello world");
        let (tag, data) = serde.dumps_typed(&input).unwrap();
        assert_eq!(tag, TAG_JSON);
        let result = serde.loads_typed(&tag, &data).unwrap();
        let output = result.downcast_ref::<JsonValue>().unwrap();
        assert_eq!(output.as_str().unwrap(), "hello world");
    }

    #[test]
    fn test_msgpack_roundtrip() {
        let input = serde_json::json!({"nested": {"data": [1, 2, 3]}});
        let bytes = to_msgpack_bytes(&input).unwrap();
        let output: JsonValue = from_msgpack_bytes(&bytes).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_unknown_tag() {
        let serde = JsonPlusSerializer::new();
        let result = serde.loads_typed("unknown", &[]);
        assert!(result.is_err());
    }
}
