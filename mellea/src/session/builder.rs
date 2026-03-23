use std::sync::Arc;

use crate::backend::{Backend, ModelOptions, OllamaBackend};
use crate::core::Context;
use crate::ivr::{RejectionSampling, SamplingStrategy};

use super::Session;

/// Builder for constructing a Session.
pub struct SessionBuilder {
    backend: Option<Arc<dyn Backend>>,
    options: ModelOptions,
    strategy: Option<Arc<dyn SamplingStrategy>>,
    system_prompt: Option<String>,
}

impl SessionBuilder {
    pub fn new() -> Self {
        Self {
            backend: None,
            options: ModelOptions::default(),
            strategy: None,
            system_prompt: None,
        }
    }

    /// Set the backend. If not called, defaults to Ollama with `granite4:micro`.
    pub fn backend(mut self, backend: impl Backend + 'static) -> Self {
        let arc: Arc<dyn Backend> = Arc::new(backend);
        self.backend = Some(arc);
        self
    }

    /// Override default model options.
    pub fn options(mut self, options: ModelOptions) -> Self {
        self.options = options;
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.options.max_tokens = Some(n);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.options.temperature = Some(t);
        self
    }

    /// Set a system prompt applied to all generations.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the sampling strategy. Defaults to `RejectionSampling { max_retries: 2 }`.
    pub fn strategy(mut self, strategy: impl SamplingStrategy + 'static) -> Self {
        let arc: Arc<dyn SamplingStrategy> = Arc::new(strategy);
        self.strategy = Some(arc);
        self
    }

    pub fn build(self) -> Session {
        let backend: Arc<dyn Backend> = self
            .backend
            .unwrap_or_else(|| Arc::new(OllamaBackend::new("granite4:micro")));

        let strategy: Arc<dyn SamplingStrategy> = self
            .strategy
            .unwrap_or_else(|| Arc::new(RejectionSampling::default()));

        let mut options = self.options;
        if let Some(prompt) = self.system_prompt {
            options.system_prompt = Some(prompt);
        }

        Session {
            backend,
            context: Context::new(),
            default_options: options,
            default_strategy: strategy,
        }
    }
}
