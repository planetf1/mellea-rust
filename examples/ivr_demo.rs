//! # IVR Demo — Instruct-Validate-Repair in action
//!
//! Demonstrates the IVR loop: the LLM generates output, requirements are checked,
//! and if validation fails, repair feedback is sent back for self-correction.
//!
//! This example intentionally uses strict requirements to trigger repair cycles,
//! showing mellea's ability to improve output quality through automated feedback.
//!
//! ## Running
//! ```sh
//! cargo run --example ivr_demo
//! cargo run --example ivr_demo -- --lmstudio
//! ```

use mellea::{Instruction, OllamaBackend, OpenAICompatBackend, RejectionSampling, Requirement, Session};
use schemars::JsonSchema;
use serde::Deserialize;

/// A structured response with tight constraints.
#[derive(Debug, Deserialize, JsonSchema)]
struct CityFact {
    /// City name
    city: String,
    /// Country the city is in
    country: String,
    /// A single interesting fact (must be under 100 characters)
    fact: String,
    /// Population estimate (must be reasonable)
    population: u64,
}

#[tokio::main]
async fn main() -> Result<(), mellea::Error> {
    let use_lmstudio = std::env::args().any(|a| a == "--lmstudio");

    let mut session = if use_lmstudio {
        Session::builder()
            .backend(OpenAICompatBackend::lmstudio("granite-4.0-h-micro"))
            .strategy(RejectionSampling::new(3)) // Allow up to 3 repair attempts
            .build()
    } else {
        Session::builder()
            .backend(OllamaBackend::new("granite4:micro"))
            .strategy(RejectionSampling::new(3))
            .build()
    };

    println!("Generating city fact with strict validation...\n");

    let result: CityFact = session
        .generate(
            Instruction::new("Tell me an interesting fact about Tokyo, Japan.")
                .response_json::<CityFact>()
                .require(
                    Requirement::new("fact must be under 100 characters")
                        .validate_with_reason(|s| {
                            // Parse JSON to check the fact field length
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                                if let Some(fact) = v.get("fact").and_then(|f| f.as_str()) {
                                    if fact.len() <= 100 {
                                        return Ok(());
                                    }
                                    return Err(format!(
                                        "fact is {} chars, must be <= 100. Shorten it.",
                                        fact.len()
                                    ));
                                }
                            }
                            Err("could not extract 'fact' field from JSON".to_string())
                        }),
                )
                .require(
                    Requirement::new("population must be between 1M and 50M")
                        .validate_with_reason(|s| {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                                if let Some(pop) = v.get("population").and_then(|p| p.as_u64()) {
                                    if (1_000_000..=50_000_000).contains(&pop) {
                                        return Ok(());
                                    }
                                    return Err(format!(
                                        "population {pop} is outside 1M-50M range"
                                    ));
                                }
                            }
                            Err("could not extract 'population' field".to_string())
                        }),
                ),
        )
        .await?;

    println!("Result (validated):");
    println!("  City:       {}", result.city);
    println!("  Country:    {}", result.country);
    println!("  Fact:       {} ({} chars)", result.fact, result.fact.len());
    println!("  Population: {}", result.population);

    Ok(())
}
