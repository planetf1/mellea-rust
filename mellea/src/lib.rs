//! # mellea — Type-Safe Generative Programming for Rust
//!
//! Mellea replaces brittle prompt engineering with **structured, validated, type-safe
//! AI workflows**. Declare what you want (a struct, an enum, a typed return value) and
//! mellea handles generation, validation, repair, and retry.
//!
//! ## Core Concepts
//!
//! - **Session** — manages conversation context and backend configuration
//! - **Instruction** — specifies what you want from the LLM, with optional requirements
//! - **Requirement** — a validation constraint checked against LLM output
//! - **IVR Loop** (Instruct-Validate-Repair) — automatic retry with error feedback
//! - **GenerativeSlot** — a typed LLM function: declare params + return type, LLM provides the body
//! - **Backend** — abstraction over LLM providers (Ollama, OpenAI-compatible)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mellea::{Session, Instruction, Requirement, OllamaBackend};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct Sentiment {
//!     label: String,
//!     confidence: f64,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mellea::Error> {
//!     let mut session = Session::builder()
//!         .backend(OllamaBackend::new("granite4:micro"))
//!         .build();
//!
//!     // Simple chat
//!     let reply = session.chat("What is Rust?").await?;
//!
//!     // Structured output with validation and automatic repair
//!     let sentiment: Sentiment = session
//!         .generate(
//!             Instruction::new("Classify the sentiment of: 'I love Rust!'")
//!                 .response_json::<Sentiment>()
//!                 .require(Requirement::new("valid label")
//!                     .validate(|s| s.contains("positive") || s.contains("negative") || s.contains("neutral")))
//!         )
//!         .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Minimum Rust Version
//!
//! Requires Rust 1.94+ (edition 2024). No backwards-compatibility shims.

pub mod backend;
pub mod component;
pub mod core;
pub mod error;
pub mod generative;
pub mod ivr;
pub mod session;

// Re-exports for ergonomic top-level imports.
pub use backend::{Backend, GenerateRequest, ModelOptions, OllamaBackend, OpenAICompatBackend};
pub use component::Instruction;
pub use core::{ChatMessage, Context, ModelOutput, Role, Turn};
pub use error::MelleaError as Error;
pub use generative::GenerativeSlot;
pub use ivr::{RejectionSampling, Requirement, SamplingStrategy, ValidationResult};
pub use session::Session;
