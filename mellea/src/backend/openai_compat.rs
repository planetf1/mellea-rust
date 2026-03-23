use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{Backend, GenerateRequest};
use crate::core::{ChatMessage, ModelOutput, Usage};
use crate::error::MelleaError;

/// Backend for OpenAI-compatible APIs (LM Studio, vLLM, llama.cpp server, etc.).
pub struct OpenAICompatBackend {
    client: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAICompatBackend {
    /// LM Studio default: localhost:1234
    pub fn lmstudio(model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: "http://localhost:1234/v1".to_string(),
            model: model.into(),
            api_key: None,
        }
    }

    /// Custom endpoint.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[derive(Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ResponseFormat {
    r#type: String,
}

#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

fn to_api_message(msg: &ChatMessage) -> ApiMessage {
    ApiMessage {
        role: serde_json::to_value(&msg.role)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "user".to_string()),
        content: msg.content.clone(),
    }
}

#[async_trait]
impl Backend for OpenAICompatBackend {
    async fn generate(&self, request: GenerateRequest) -> Result<ModelOutput, MelleaError> {
        let messages: Vec<ApiMessage> = request.messages.iter().map(to_api_message).collect();

        let response_format = if request.json_schema.is_some() {
            Some(ResponseFormat {
                r#type: "json_object".to_string(),
            })
        } else {
            None
        };

        let body = CompletionRequest {
            model: self.model.clone(),
            messages,
            max_tokens: request.options.max_tokens,
            temperature: request.options.temperature,
            seed: request.options.seed,
            stop: request.options.stop.clone(),
            response_format,
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.client.post(&url).json(&body);

        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: CompletionResponse = req
            .send()
            .await?
            .error_for_status()
            .map_err(|e| MelleaError::Backend(e.to_string()))?
            .json()
            .await?;

        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = resp.usage.map_or(Usage::default(), |u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(ModelOutput {
            content,
            model: self.model.clone(),
            usage,
        })
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}
