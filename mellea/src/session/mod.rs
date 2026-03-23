mod builder;

pub use builder::SessionBuilder;

use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::backend::{Backend, GenerateRequest, ModelOptions};
use crate::component::Instruction;
use crate::core::{ChatMessage, Context, ModelOutput, Turn};
use crate::error::MelleaError;
use crate::ivr::SamplingStrategy;

/// The primary user-facing API for mellea.
///
/// Manages conversation context, backend configuration, and the IVR loop.
/// Each generation updates the internal context, threading the conversation.
pub struct Session {
    backend: Arc<dyn Backend>,
    context: Context,
    default_options: ModelOptions,
    default_strategy: Arc<dyn SamplingStrategy>,
}

impl Session {
    pub fn builder() -> SessionBuilder {
        SessionBuilder::new()
    }

    /// Simple chat: send a message, get a response string.
    pub async fn chat(&mut self, content: &str) -> Result<String, MelleaError> {
        let mut messages = self.context.messages();
        messages.push(ChatMessage::user(content));

        let request = GenerateRequest {
            messages,
            options: self.default_options.clone(),
            json_schema: None,
        };

        let output = self.backend.generate(request).await?;
        let reply = output.content.clone();

        self.context = self.context.push(Turn {
            input: ChatMessage::user(content),
            output: Some(output),
        });

        Ok(reply)
    }

    /// Instruct: send an Instruction, run the IVR loop, return the ModelOutput.
    pub async fn instruct(
        &mut self,
        instruction: Instruction,
    ) -> Result<ModelOutput, MelleaError> {
        let mut messages = self.context.messages();
        let inst_messages = instruction.to_messages();
        messages.extend(inst_messages.clone());

        let request = GenerateRequest {
            messages,
            options: self.default_options.clone(),
            json_schema: instruction.json_schema.clone(),
        };

        let result = self
            .default_strategy
            .sample(self.backend.as_ref(), request, &instruction.requirements)
            .await?;

        let input_content = inst_messages
            .into_iter()
            .map(|m| m.content)
            .collect::<Vec<_>>()
            .join("\n");

        self.context = self.context.push(Turn {
            input: ChatMessage::user(input_content),
            output: Some(result.output.clone()),
        });

        Ok(result.output)
    }

    /// Generate a typed value: send an Instruction, run IVR, parse into T.
    ///
    /// Automatically adds a `parses_as::<T>()` requirement so the IVR loop
    /// will retry if the LLM returns malformed JSON or missing fields.
    pub async fn generate<T>(&mut self, mut instruction: Instruction) -> Result<T, MelleaError>
    where
        T: DeserializeOwned + JsonSchema + 'static,
    {
        // Auto-add parse requirement if not already present, so the IVR loop
        // catches JSON parse failures and retries with repair feedback.
        let has_parse_req = instruction.requirements.iter().any(|r| {
            r.description.contains("parses as") || r.description.contains("valid JSON")
        });
        if !has_parse_req {
            instruction = instruction.require(
                crate::ivr::Requirement::new(
                    format!("output must be valid JSON that parses as {}", std::any::type_name::<T>())
                )
                .parses_as::<T>(),
            );
        }

        let output = self.instruct(instruction).await?;
        output.parse()
    }

    /// Access the current conversation context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Access the backend (used by GenerativeSlot).
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Push a turn into the context (used by GenerativeSlot).
    pub fn push_turn(&mut self, input: ChatMessage, output: ModelOutput) {
        self.context = self.context.push(Turn {
            input,
            output: Some(output),
        });
    }

    /// Reset conversation context.
    pub fn reset(&mut self) {
        self.context = Context::new();
    }
}
