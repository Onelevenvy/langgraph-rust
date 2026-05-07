use std::any::Any;
use crate::error::SerdeError;

/// Trait for typed serialization. Mirrors Python's SerializerProtocol.
pub trait SerializerProtocol: Send + Sync {
    /// Serialize an object to a (type_tag, bytes) pair.
    fn dumps_typed(&self, obj: &dyn Any) -> Result<(String, Vec<u8>), SerdeError>;

    /// Deserialize from a (type_tag, bytes) pair.
    fn loads_typed(&self, tag: &str, data: &[u8]) -> Result<Box<dyn Any>, SerdeError>;
}

/// Untyped serialization protocol.
pub trait UntypedSerializerProtocol: Send + Sync {
    fn dumps(&self, obj: &dyn Any) -> Result<Vec<u8>, SerdeError>;
    fn loads(&self, data: &[u8]) -> Result<Box<dyn Any>, SerdeError>;
}

/// Cipher protocol for encrypted serialization.
pub trait CipherProtocol: Send + Sync {
    fn encrypt(&self, plaintext: &[u8]) -> Result<(String, Vec<u8>), SerdeError>;
    fn decrypt(&self, cipher_name: &str, ciphertext: &[u8]) -> Result<Vec<u8>, SerdeError>;
}
