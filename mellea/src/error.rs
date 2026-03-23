use thiserror::Error;

#[derive(Error, Debug)]
pub enum MelleaError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("parse error: could not parse LLM output as {target_type}: {reason}")]
    Parse {
        target_type: &'static str,
        reason: String,
        raw_output: String,
    },

    #[error("validation failed after {attempts} attempt(s): {reason}")]
    ValidationFailed {
        attempts: u32,
        reason: String,
        history: Vec<ValidationAttempt>,
    },

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("ollama error: {0}")]
    Ollama(String),

    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ValidationAttempt {
    pub raw_output: String,
    pub failures: Vec<ValidationFailure>,
}

#[derive(Debug, Clone)]
pub struct ValidationFailure {
    pub requirement: String,
    pub reason: String,
}
