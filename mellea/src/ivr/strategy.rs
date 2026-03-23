use async_trait::async_trait;

use super::requirement::{Requirement, ValidationResult};
use crate::backend::{Backend, GenerateRequest};
use crate::core::{ChatMessage, ModelOutput};
use crate::error::{MelleaError, ValidationAttempt, ValidationFailure};

/// The result of a sampling strategy run.
#[derive(Debug)]
pub struct SamplingResult {
    /// The final (best) output.
    pub output: ModelOutput,
    /// History of all attempts.
    pub attempts: Vec<SamplingAttempt>,
    /// Whether validation succeeded.
    pub success: bool,
}

/// A single generation attempt with its validation results.
#[derive(Debug, Clone)]
pub struct SamplingAttempt {
    pub output: ModelOutput,
    pub validations: Vec<(String, ValidationResult)>,
}

/// Strategy for generating and validating LLM output.
#[async_trait]
pub trait SamplingStrategy: Send + Sync {
    async fn sample(
        &self,
        backend: &dyn Backend,
        request: GenerateRequest,
        requirements: &[Requirement],
    ) -> Result<SamplingResult, MelleaError>;
}

/// Default strategy: generate, validate, repair with feedback, retry.
///
/// On validation failure, appends a repair message containing the failure reasons
/// and asks the LLM to correct its output. Retries up to `max_retries` times.
pub struct RejectionSampling {
    pub max_retries: u32,
}

impl Default for RejectionSampling {
    fn default() -> Self {
        Self { max_retries: 2 }
    }
}

impl RejectionSampling {
    pub fn new(max_retries: u32) -> Self {
        Self { max_retries }
    }

    fn build_repair_message(failures: &[(String, ValidationResult)]) -> String {
        let mut msg = String::from(
            "Your previous response did not meet the requirements. Please correct it.\n\nFailures:\n",
        );
        for (desc, result) in failures {
            if let ValidationResult::Fail { reason } = result {
                msg.push_str(&format!("- {desc}: {reason}\n"));
            }
        }
        msg.push_str("\nPlease provide a corrected response that satisfies all requirements.");
        msg
    }
}

#[async_trait]
impl SamplingStrategy for RejectionSampling {
    async fn sample(
        &self,
        backend: &dyn Backend,
        request: GenerateRequest,
        requirements: &[Requirement],
    ) -> Result<SamplingResult, MelleaError> {
        let mut attempts = Vec::new();
        let mut current_messages = request.messages.clone();

        for attempt_num in 0..=self.max_retries {
            let mut attempt_request = request.clone();
            attempt_request.messages = current_messages.clone();

            let output = backend.generate(attempt_request).await?;

            // Validate against all requirements
            let validations: Vec<(String, ValidationResult)> = requirements
                .iter()
                .map(|req| (req.description.clone(), req.check(&output.content)))
                .collect();

            let all_pass = validations.iter().all(|(_, v)| v.is_pass());

            let attempt = SamplingAttempt {
                output: output.clone(),
                validations: validations.clone(),
            };
            attempts.push(attempt);

            if all_pass {
                return Ok(SamplingResult {
                    output,
                    attempts,
                    success: true,
                });
            }

            // If we have retries left, add repair context
            if attempt_num < self.max_retries {
                let failures: Vec<_> =
                    validations.iter().filter(|(_, v)| v.is_fail()).cloned().collect();

                // Add the LLM's failed output and our repair message
                current_messages.push(ChatMessage::assistant(&output.content));
                current_messages.push(ChatMessage::user(Self::build_repair_message(&failures)));

                tracing::debug!(
                    attempt = attempt_num + 1,
                    max = self.max_retries,
                    failures = failures.len(),
                    "IVR: validation failed, sending repair feedback"
                );
            }
        }

        // Budget exhausted — return the last attempt with error
        let last = attempts.last().unwrap();
        let failure_reasons: Vec<String> = last
            .validations
            .iter()
            .filter_map(|(desc, v)| {
                v.reason().map(|r| format!("{desc}: {r}"))
            })
            .collect();

        Err(MelleaError::ValidationFailed {
            attempts: self.max_retries + 1,
            reason: failure_reasons.join("; "),
            history: attempts
                .iter()
                .map(|a| ValidationAttempt {
                    raw_output: a.output.content.clone(),
                    failures: a
                        .validations
                        .iter()
                        .filter_map(|(desc, v)| {
                            v.reason().map(|r| ValidationFailure {
                                requirement: desc.clone(),
                                reason: r.to_string(),
                            })
                        })
                        .collect(),
                })
                .collect(),
        })
    }
}
