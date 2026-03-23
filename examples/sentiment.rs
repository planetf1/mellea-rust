//! # Sentiment Classifier — Structured output with type safety
//!
//! Demonstrates mellea's core value: declare a Rust struct, get that struct back from the LLM.
//! The JSON schema is generated from `#[derive(JsonSchema)]` and injected into the prompt.
//! If the LLM returns invalid JSON, mellea's IVR loop repairs it automatically.
//!
//! ## Running
//! ```sh
//! cargo run --example sentiment
//! cargo run --example sentiment -- --lmstudio
//! ```

use mellea::{Instruction, OllamaBackend, OpenAICompatBackend, Requirement, Session};
use schemars::JsonSchema;
use serde::Deserialize;

/// The LLM must return this exact structure.
#[derive(Debug, Deserialize, JsonSchema)]
struct Sentiment {
    /// One of: positive, negative, neutral
    label: String,
    /// Confidence score between 0.0 and 1.0
    confidence: f64,
}

#[tokio::main]
async fn main() -> Result<(), mellea::Error> {
    let use_lmstudio = std::env::args().any(|a| a == "--lmstudio");

    let mut session = if use_lmstudio {
        Session::builder()
            .backend(OpenAICompatBackend::lmstudio("granite-4.0-h-micro"))
            .build()
    } else {
        Session::builder()
            .backend(OllamaBackend::new("granite4:micro"))
            .build()
    };

    let texts = [
        "I absolutely love writing Rust code!",
        "This library is terrible and wastes my time.",
        "The function returned a value.",
    ];

    for text in texts {
        let sentiment: Sentiment = session
            .generate(
                Instruction::new(format!("Classify the sentiment of: '{text}'"))
                    .response_json::<Sentiment>()
                    .require(
                        Requirement::new("label must be positive, negative, or neutral")
                            .validate(|s| {
                                s.contains("positive")
                                    || s.contains("negative")
                                    || s.contains("neutral")
                            }),
                    ),
            )
            .await?;

        println!("{text}");
        println!("  -> {} (confidence: {:.2})\n", sentiment.label, sentiment.confidence);

        // Reset context between classifications so they're independent
        session.reset();
    }

    Ok(())
}
