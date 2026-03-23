use serde::de::DeserializeOwned;

use crate::error::MelleaError;

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub content: String,
    pub model: String,
    pub usage: Usage,
}

impl ModelOutput {
    /// Attempt to parse the content as JSON into type T.
    /// Strips common LLM artifacts (markdown fences, leading text) before parsing.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T, MelleaError> {
        let cleaned = strip_json_artifacts(&self.content);
        serde_json::from_str::<T>(&cleaned).map_err(|e| MelleaError::Parse {
            target_type: std::any::type_name::<T>(),
            reason: e.to_string(),
            raw_output: self.content.clone(),
        })
    }
}

/// Strip markdown code fences and leading/trailing noise from LLM JSON output.
pub fn strip_json_artifacts(s: &str) -> String {
    let trimmed = s.trim();

    // Strip ```json ... ``` fences
    if let Some(rest) = trimmed.strip_prefix("```json") {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim().to_string();
        }
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim().to_string();
        }
    }

    // Try to find the first { or [ and last } or ]
    if let (Some(start), Some(end)) = (find_json_start(trimmed), find_json_end(trimmed)) {
        if start <= end {
            return trimmed[start..=end].to_string();
        }
    }

    trimmed.to_string()
}

fn find_json_start(s: &str) -> Option<usize> {
    s.find('{').or_else(|| s.find('['))
}

fn find_json_end(s: &str) -> Option<usize> {
    s.rfind('}').or_else(|| s.rfind(']'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn test_strip_json_fences() {
        assert_eq!(strip_json_artifacts("```json\n{\"a\": 1}\n```"), "{\"a\": 1}");
        assert_eq!(strip_json_artifacts("```\n{\"a\": 1}\n```"), "{\"a\": 1}");
    }

    #[test]
    fn test_strip_leading_text() {
        let input = "Here is the result:\n{\"label\": \"positive\"}";
        assert_eq!(strip_json_artifacts(input), "{\"label\": \"positive\"}");
    }

    #[test]
    fn test_clean_json_passthrough() {
        assert_eq!(strip_json_artifacts("{\"a\": 1}"), "{\"a\": 1}");
    }

    #[test]
    fn test_parse_model_output() {
        #[derive(Deserialize, Debug, PartialEq)]
        struct S {
            label: String,
        }
        let output = ModelOutput {
            content: "{\"label\": \"positive\"}".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
        };
        let parsed: S = output.parse().unwrap();
        assert_eq!(parsed, S { label: "positive".to_string() });
    }

    #[test]
    fn test_parse_with_fences() {
        #[derive(Deserialize, Debug, PartialEq)]
        struct S {
            x: i32,
        }
        let output = ModelOutput {
            content: "```json\n{\"x\": 42}\n```".to_string(),
            model: "test".to_string(),
            usage: Usage::default(),
        };
        let parsed: S = output.parse().unwrap();
        assert_eq!(parsed, S { x: 42 });
    }
}
