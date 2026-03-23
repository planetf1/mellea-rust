use std::marker::PhantomData;

use schemars::JsonSchema;
use schemars::schema_for;
use serde::de::DeserializeOwned;

use crate::backend::{GenerateRequest, ModelOptions};
use crate::component::Instruction;
use crate::core::ChatMessage;
use crate::error::MelleaError;
use crate::ivr::{Requirement, RejectionSampling, SamplingStrategy};
use crate::session::Session;

/// A parameter definition for a generative function.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: String,
    pub type_name: String,
    pub description: String,
}

/// A generative function: a typed LLM call with a name, description, parameters,
/// and a return type. The LLM provides the implementation.
///
/// Built via `GenerativeSlot::<T>::builder()`.
pub struct GenerativeSlot<T: DeserializeOwned + JsonSchema> {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamDef>,
    pub requirements: Vec<Requirement>,
    pub max_retries: u32,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema> std::fmt::Debug for GenerativeSlot<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerativeSlot")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("params", &self.params)
            .finish()
    }
}

impl<T: DeserializeOwned + JsonSchema + Send + 'static> GenerativeSlot<T> {
    pub fn builder() -> GenerativeSlotBuilder<T> {
        GenerativeSlotBuilder {
            name: String::new(),
            description: String::new(),
            params: Vec::new(),
            requirements: Vec::new(),
            max_retries: 2,
            _phantom: PhantomData,
        }
    }

    /// Call this generative function with the given arguments.
    /// Uses the session's backend and applies the IVR loop.
    pub async fn call(
        &self,
        session: &mut Session,
        args: serde_json::Value,
    ) -> Result<T, MelleaError> {
        let instruction = self.build_instruction(&args);
        let messages = {
            let mut msgs = session.context().messages();
            msgs.extend(instruction.to_messages());
            msgs
        };

        let schema = schema_for!(T);
        let json_schema = serde_json::to_value(&schema).ok();

        let request = GenerateRequest {
            messages,
            options: ModelOptions::default(),
            json_schema,
        };

        let strategy = RejectionSampling::new(self.max_retries);
        let result = strategy
            .sample(session.backend(), request, &instruction.requirements)
            .await?;

        let parsed: T = result.output.parse()?;

        // Update session context with this turn
        session.push_turn(
            ChatMessage::user(format!("{}({})", self.name, args)),
            result.output,
        );

        Ok(parsed)
    }

    fn build_instruction(&self, args: &serde_json::Value) -> Instruction {
        let mut desc = format!("You are implementing the function `{}`.\n\n", self.name);
        desc.push_str(&self.description);

        if !self.params.is_empty() {
            desc.push_str("\n\n## Parameters\n");
            for param in &self.params {
                let val = args.get(&param.name).map_or("(not provided)".to_string(), |v| v.to_string());
                desc.push_str(&format!("- `{}` ({}): {} = {}\n", param.name, param.type_name, param.description, val));
            }
        }

        let mut inst = Instruction::new(desc).response_json::<T>();

        // Always add a parse requirement so the IVR loop catches JSON errors
        inst = inst.require(
            Requirement::new(format!(
                "output must be valid JSON that parses as {}",
                std::any::type_name::<T>()
            ))
            .parses_as::<T>(),
        );

        // Add user-specified requirements (rebuilt as parse checks since closures aren't Clone)
        for req in &self.requirements {
            inst = inst.require(
                Requirement::new(&req.description).parses_as::<T>(),
            );
        }
        inst
    }
}

/// Builder for GenerativeSlot.
pub struct GenerativeSlotBuilder<T: DeserializeOwned + JsonSchema> {
    name: String,
    description: String,
    params: Vec<ParamDef>,
    requirements: Vec<Requirement>,
    max_retries: u32,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned + JsonSchema + Send + 'static> GenerativeSlotBuilder<T> {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a parameter definition.
    pub fn param<P: 'static>(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.params.push(ParamDef {
            name: name.into(),
            type_name: std::any::type_name::<P>().to_string(),
            description: description.into(),
        });
        self
    }

    pub fn require(mut self, req: Requirement) -> Self {
        self.requirements.push(req);
        self
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn build(self) -> GenerativeSlot<T> {
        GenerativeSlot {
            name: self.name,
            description: self.description,
            params: self.params,
            requirements: self.requirements,
            max_retries: self.max_retries,
            _phantom: PhantomData,
        }
    }
}
