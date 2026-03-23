//! # Generative Function — LLM-powered typed functions
//!
//! Demonstrates mellea's generative function abstraction: define a function signature
//! with typed parameters and return type, and the LLM provides the implementation.
//!
//! This is the "generative programming" paradigm — treating LLM calls as typed function
//! calls that integrate naturally into Rust code.
//!
//! ## Running
//! ```sh
//! cargo run --example generative_fn
//! cargo run --example generative_fn -- --lmstudio
//! ```

use mellea::{GenerativeSlot, OllamaBackend, OpenAICompatBackend, Session};
use schemars::JsonSchema;
use serde::Deserialize;

/// Named entities extracted from text.
#[derive(Debug, Deserialize, JsonSchema)]
struct Entity {
    name: String,
    /// One of: person, organization, location, other
    kind: String,
}

/// The return type of the extraction function.
#[derive(Debug, Deserialize, JsonSchema)]
struct Entities {
    entities: Vec<Entity>,
}

/// A translation result.
#[derive(Debug, Deserialize, JsonSchema)]
struct Translation {
    original: String,
    translated: String,
    target_language: String,
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

    // --- Generative function: Entity Extraction ---
    let extract_entities = GenerativeSlot::<Entities>::builder()
        .name("extract_entities")
        .description("Extract all named entities (people, organizations, locations) from text")
        .param::<String>("text", "The text to extract entities from")
        .build();

    println!("=== Entity Extraction ===\n");
    let result = extract_entities
        .call(
            &mut session,
            serde_json::json!({"text": "Tim Cook announced new Apple products at WWDC in Cupertino, California."}),
        )
        .await?;

    for entity in &result.entities {
        println!("  {} ({})", entity.name, entity.kind);
    }

    session.reset();

    // --- Generative function: Translation ---
    let translate = GenerativeSlot::<Translation>::builder()
        .name("translate")
        .description("Translate text to the specified target language")
        .param::<String>("text", "The text to translate")
        .param::<String>("target_language", "The language to translate into")
        .build();

    println!("\n=== Translation ===\n");
    let result = translate
        .call(
            &mut session,
            serde_json::json!({
                "text": "The quick brown fox jumps over the lazy dog",
                "target_language": "French"
            }),
        )
        .await?;

    println!("  Original: {}", result.original);
    println!("  Translated: {}", result.translated);
    println!("  Language: {}", result.target_language);

    Ok(())
}
