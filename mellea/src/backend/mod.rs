mod ollama;
mod openai_compat;

pub use ollama::OllamaBackend;
pub use openai_compat::OpenAICompatBackend;

use async_trait::async_trait;

use crate::core::{ChatMessage, ModelOutput};
use crate::error::MelleaError;

/// Options controlling LLM generation behavior.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<Vec<String>>,
    pub system_prompt: Option<String>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            seed: None,
            stop: None,
            system_prompt: None,
        }
    }
}

/// Request to generate a completion.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub options: ModelOptions,
    /// Optional JSON schema the response must conform to.
    pub json_schema: Option<serde_json::Value>,
}

/// The core abstraction over LLM providers.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Generate a completion from a sequence of messages.
    async fn generate(&self, request: GenerateRequest) -> Result<ModelOutput, MelleaError>;

    /// The model identifier this backend is configured for.
    fn model_id(&self) -> &str;
}
