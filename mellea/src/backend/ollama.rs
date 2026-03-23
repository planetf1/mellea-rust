use async_trait::async_trait;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::ChatMessage as OllamaChatMessage;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::models::ModelOptions as OllamaModelOptions;

use super::{Backend, GenerateRequest, ModelOptions};
use crate::core::{ChatMessage, ModelOutput, Role};
use crate::error::MelleaError;

/// Ollama backend using the `ollama-rs` crate.
///
/// Connects to a local or remote Ollama instance and generates completions
/// via the chat API. Default: `localhost:11434`.
pub struct OllamaBackend {
    client: Ollama,
    model: String,
}

impl OllamaBackend {
    /// Create with default host (`localhost:11434`).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Ollama::default(),
            model: model.into(),
        }
    }

    /// Create with a custom host URL and port.
    pub fn with_host(host: &str, port: u16, model: impl Into<String>) -> Self {
        Self {
            client: Ollama::new(host, port),
            model: model.into(),
        }
    }
}

fn to_ollama_message(msg: &ChatMessage) -> OllamaChatMessage {
    match msg.role {
        Role::System => OllamaChatMessage::system(msg.content.clone()),
        Role::User => OllamaChatMessage::user(msg.content.clone()),
        Role::Assistant => OllamaChatMessage::assistant(msg.content.clone()),
        Role::Tool => OllamaChatMessage::user(msg.content.clone()),
    }
}

fn build_model_options(opts: &ModelOptions) -> OllamaModelOptions {
    let mut mopts = OllamaModelOptions::default();
    if let Some(temp) = opts.temperature {
        mopts = mopts.temperature(temp);
    }
    if let Some(max) = opts.max_tokens {
        mopts = mopts.num_predict(max as i32);
    }
    if let Some(seed) = opts.seed {
        mopts = mopts.seed(seed as i32);
    }
    if let Some(ref stop) = opts.stop {
        mopts = mopts.stop(stop.clone());
    }
    mopts
}

#[async_trait]
impl Backend for OllamaBackend {
    async fn generate(&self, request: GenerateRequest) -> Result<ModelOutput, MelleaError> {
        let messages: Vec<OllamaChatMessage> =
            request.messages.iter().map(to_ollama_message).collect();

        let mut chat_req = ChatMessageRequest::new(self.model.clone(), messages);
        chat_req = chat_req.options(build_model_options(&request.options));

        if request.json_schema.is_some() {
            chat_req = chat_req.format(ollama_rs::generation::parameters::FormatType::Json);
        }

        let response = self
            .client
            .send_chat_messages(chat_req)
            .await
            .map_err(|e| MelleaError::Ollama(e.to_string()))?;

        Ok(ModelOutput {
            content: response.message.content,
            model: self.model.clone(),
            usage: crate::core::Usage::default(),
        })
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
