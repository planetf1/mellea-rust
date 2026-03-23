use serde::de::DeserializeOwned;
use std::marker::PhantomData;

/// The outcome of checking a requirement against LLM output.
#[derive(Debug, Clone)]
pub enum ValidationResult {
    Pass,
    Fail { reason: String },
}

impl ValidationResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail { .. })
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Fail { reason } => Some(reason),
            Self::Pass => None,
        }
    }
}

/// A validation constraint applied to LLM output.
/// Requirements have a human-readable description (included in repair prompts)
/// and a validation function that checks raw output.
pub struct Requirement {
    pub description: String,
    validate_fn: Box<dyn Fn(&str) -> ValidationResult + Send + Sync>,
}

impl std::fmt::Debug for Requirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Requirement")
            .field("description", &self.description)
            .finish()
    }
}

impl Requirement {
    /// Start building a requirement with a description.
    pub fn new(description: impl Into<String>) -> RequirementBuilder {
        RequirementBuilder {
            description: description.into(),
        }
    }

    /// Check this requirement against raw LLM output.
    pub fn check(&self, output: &str) -> ValidationResult {
        (self.validate_fn)(output)
    }
}

/// Builder for constructing requirements with different validation strategies.
pub struct RequirementBuilder {
    description: String,
}

impl RequirementBuilder {
    /// Validate with a closure that receives the raw output string.
    pub fn validate(self, f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Requirement {
        let desc = self.description.clone();
        Requirement {
            description: self.description,
            validate_fn: Box::new(move |output| {
                if f(output) {
                    ValidationResult::Pass
                } else {
                    ValidationResult::Fail {
                        reason: format!("requirement not met: {desc}"),
                    }
                }
            }),
        }
    }

    /// Validate with a closure that returns Ok(()) on success or Err(reason) on failure.
    pub fn validate_with_reason(
        self,
        f: impl Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    ) -> Requirement {
        Requirement {
            description: self.description,
            validate_fn: Box::new(move |output| match f(output) {
                Ok(()) => ValidationResult::Pass,
                Err(reason) => ValidationResult::Fail { reason },
            }),
        }
    }

    /// Validate that the output parses as JSON into type T.
    pub fn parses_as<T: DeserializeOwned + 'static>(self) -> Requirement {
        let desc = self.description.clone();
        Requirement {
            description: self.description,
            validate_fn: Box::new(move |output| {
                let cleaned = crate::core::output::strip_json_artifacts(output);
                match serde_json::from_str::<T>(&cleaned) {
                    Ok(_) => ValidationResult::Pass,
                    Err(e) => ValidationResult::Fail {
                        reason: format!("{desc}: JSON parse error: {e}"),
                    },
                }
            }),
        }
    }

    /// Validate that output is at most n characters.
    pub fn max_length(self, n: usize) -> Requirement {
        Requirement {
            description: self.description,
            validate_fn: Box::new(move |output| {
                if output.len() <= n {
                    ValidationResult::Pass
                } else {
                    ValidationResult::Fail {
                        reason: format!("output length {} exceeds max {n}", output.len()),
                    }
                }
            }),
        }
    }

    /// Validate that output is at least n characters.
    pub fn min_length(self, n: usize) -> Requirement {
        Requirement {
            description: self.description,
            validate_fn: Box::new(move |output| {
                if output.len() >= n {
                    ValidationResult::Pass
                } else {
                    ValidationResult::Fail {
                        reason: format!("output length {} below min {n}", output.len()),
                    }
                }
            }),
        }
    }
}

/// Validate a typed value after parsing.
/// Intended for post-parse validation in future typed pipelines.
#[allow(dead_code)]
pub struct TypedRequirement<T> {
    pub description: String,
    validate_fn: Box<dyn Fn(&T) -> ValidationResult + Send + Sync>,
    _phantom: PhantomData<T>,
}

impl<T> std::fmt::Debug for TypedRequirement<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedRequirement")
            .field("description", &self.description)
            .finish()
    }
}

#[allow(dead_code)]
impl<T: 'static> TypedRequirement<T> {
    pub fn new(
        description: impl Into<String>,
        f: impl Fn(&T) -> bool + Send + Sync + 'static,
    ) -> Self {
        let desc: String = description.into();
        let desc_clone = desc.clone();
        Self {
            description: desc,
            validate_fn: Box::new(move |val| {
                if f(val) {
                    ValidationResult::Pass
                } else {
                    ValidationResult::Fail {
                        reason: format!("typed requirement not met: {desc_clone}"),
                    }
                }
            }),
            _phantom: PhantomData,
        }
    }

    pub fn check(&self, value: &T) -> ValidationResult {
        (self.validate_fn)(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_closure() {
        let req = Requirement::new("must contain 'hello'").validate(|s| s.contains("hello"));
        assert!(req.check("hello world").is_pass());
        assert!(req.check("goodbye").is_fail());
    }

    #[test]
    fn test_validate_with_reason() {
        let req = Requirement::new("must be numeric").validate_with_reason(|s| {
            s.trim()
                .parse::<f64>()
                .map(|_| ())
                .map_err(|e| format!("not a number: {e}"))
        });
        assert!(req.check("42").is_pass());
        assert!(req.check("abc").is_fail());
    }

    #[test]
    fn test_parses_as() {
        #[derive(serde::Deserialize)]
        struct Foo {
            x: i32,
        }
        let req = Requirement::new("must be valid Foo JSON").parses_as::<Foo>();
        assert!(req.check(r#"{"x": 1}"#).is_pass());
        assert!(req.check(r#"{"y": 1}"#).is_fail());
        assert!(req.check("not json").is_fail());
    }

    #[test]
    fn test_max_length() {
        let req = Requirement::new("short output").max_length(10);
        assert!(req.check("hi").is_pass());
        assert!(req.check("this is way too long").is_fail());
    }

    #[test]
    fn test_typed_requirement() {
        let req = TypedRequirement::new("x must be positive", |val: &i32| *val > 0);
        assert!(req.check(&5).is_pass());
        assert!(req.check(&-1).is_fail());
    }
}
