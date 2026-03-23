use std::collections::HashMap;

use schemars::JsonSchema;
use schemars::schema_for;

use crate::core::ChatMessage;
use crate::ivr::Requirement;

/// An in-context learning example: input/output pair.
#[derive(Debug, Clone)]
pub struct Example {
    pub input: String,
    pub output: String,
}

/// The primary component for specifying what you want from the LLM.
///
/// Combines a task description with optional requirements (for IVR validation),
/// grounding context, in-context examples, and response format constraints.
#[derive(Debug)]
pub struct Instruction {
    pub description: String,
    pub requirements: Vec<Requirement>,
    pub grounding: HashMap<String, String>,
    pub examples: Vec<Example>,
    pub json_schema: Option<serde_json::Value>,
    schema_type_name: Option<String>,
}

impl Instruction {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            requirements: Vec::new(),
            grounding: HashMap::new(),
            examples: Vec::new(),
            json_schema: None,
            schema_type_name: None,
        }
    }

    /// Add a validation requirement.
    pub fn require(mut self, req: Requirement) -> Self {
        self.requirements.push(req);
        self
    }

    /// Add grounding context (key-value pairs included in the prompt).
    pub fn ground(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.grounding.insert(key.into(), value.into());
        self
    }

    /// Add an in-context learning example.
    pub fn example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push(Example {
            input: input.into(),
            output: output.into(),
        });
        self
    }

    /// Require JSON output conforming to the schema of type T.
    /// Generates a JSON Schema from T's `JsonSchema` implementation and injects it into the prompt.
    pub fn response_json<T: JsonSchema>(mut self) -> Self {
        let schema = schema_for!(T);
        self.json_schema = Some(serde_json::to_value(&schema).unwrap_or_default());
        self.schema_type_name = Some(std::any::type_name::<T>().to_string());
        self
    }

    /// Format this instruction into ChatMessages suitable for an API call.
    pub fn to_messages(&self) -> Vec<ChatMessage> {
        let mut parts = Vec::new();

        // Main instruction
        parts.push(self.description.clone());

        // Grounding context
        if !self.grounding.is_empty() {
            parts.push("\n## Context".to_string());
            for (key, value) in &self.grounding {
                parts.push(format!("### {key}\n{value}"));
            }
        }

        // In-context examples
        if !self.examples.is_empty() {
            parts.push("\n## Examples".to_string());
            for (i, ex) in self.examples.iter().enumerate() {
                parts.push(format!("### Example {}\nInput: {}\nOutput: {}", i + 1, ex.input, ex.output));
            }
        }

        // Requirements (human-readable, so the LLM knows what to satisfy)
        if !self.requirements.is_empty() {
            parts.push("\n## Requirements".to_string());
            for req in &self.requirements {
                parts.push(format!("- {}", req.description));
            }
        }

        // JSON schema constraint — provide a compact example shape rather than full schema,
        // as small models tend to echo verbose schemas instead of generating a response.
        if let Some(ref schema) = self.json_schema {
            let example = build_example_from_schema(schema);
            parts.push(format!(
                "\nRespond with ONLY valid JSON. Example format:\n{example}\nOutput raw JSON only. No other text, no markdown, no explanation."
            ));
        }

        vec![ChatMessage::user(parts.join("\n"))]
    }
}

/// Build a compact example JSON object from a JSON Schema.
/// Recurses into nested objects and array items so the model sees the full shape.
fn build_example_from_schema(schema: &serde_json::Value) -> String {
    let value = schema_to_example(schema, schema);
    serde_json::to_string_pretty(&value).unwrap_or_default()
}

/// Recursively convert a JSON Schema node into an example JSON value.
/// `root` is the top-level schema (for resolving `$ref` / `definitions`).
fn schema_to_example(node: &serde_json::Value, root: &serde_json::Value) -> serde_json::Value {
    // Handle $ref
    if let Some(ref_path) = node.get("$ref").and_then(|r| r.as_str()) {
        if let Some(def_name) = ref_path.strip_prefix("#/definitions/") {
            if let Some(def) = root.get("definitions").and_then(|d| d.get(def_name)) {
                return schema_to_example(def, root);
            }
        }
        return serde_json::json!({});
    }

    let type_str = node.get("type").and_then(|t| t.as_str()).unwrap_or("object");
    let desc = node.get("description").and_then(|d| d.as_str());

    match type_str {
        "object" => {
            let mut fields = serde_json::Map::new();
            if let Some(props) = node.get("properties").and_then(|p| p.as_object()) {
                for (name, prop) in props {
                    fields.insert(name.clone(), schema_to_example(prop, root));
                }
            }
            serde_json::Value::Object(fields)
        }
        "array" => {
            let item_example = node
                .get("items")
                .map(|items| schema_to_example(items, root))
                .unwrap_or(serde_json::json!("..."));
            serde_json::json!([item_example])
        }
        "number" => {
            if let Some(d) = desc {
                if d.to_lowercase().contains("0.0")
                    || d.to_lowercase().contains("1.0")
                    || d.to_lowercase().contains("confidence")
                {
                    return serde_json::json!(0.85);
                }
            }
            serde_json::json!(0)
        }
        "integer" => serde_json::json!(0),
        "boolean" => serde_json::json!(true),
        _ => {
            if let Some(d) = desc {
                serde_json::Value::String(format!("<{d}>"))
            } else {
                serde_json::Value::String("...".to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_basic() {
        let inst = Instruction::new("Do something");
        let msgs = inst.to_messages();
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].content.contains("Do something"));
    }

    #[test]
    fn test_instruction_with_requirements() {
        let inst = Instruction::new("Classify sentiment")
            .require(
                crate::ivr::Requirement::new("must be positive or negative")
                    .validate(|s| s.contains("positive") || s.contains("negative")),
            )
            .ground("text", "I love Rust");

        let msgs = inst.to_messages();
        assert!(msgs[0].content.contains("Requirements"));
        assert!(msgs[0].content.contains("must be positive or negative"));
        assert!(msgs[0].content.contains("I love Rust"));
    }

    #[test]
    fn test_instruction_json_schema() {
        #[derive(schemars::JsonSchema)]
        struct Mood {
            label: String,
        }
        let inst = Instruction::new("Get mood").response_json::<Mood>();
        let msgs = inst.to_messages();
        assert!(msgs[0].content.contains("JSON"));
        assert!(msgs[0].content.contains("label"));
    }
}
