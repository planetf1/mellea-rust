//! Integration tests for mellea.
//!
//! These tests require a running Ollama instance on localhost:11434
//! with the `granite4:micro` model pulled.
//!
//! Run with: `cargo test --test integration`
//! Skip with: `cargo test --lib` (unit tests only)

use mellea::{Instruction, OllamaBackend, Requirement, Session};
use schemars::JsonSchema;
use serde::Deserialize;

/// Check if Ollama is reachable before running tests.
async fn ollama_available() -> bool {
    reqwest::get("http://localhost:11434/api/tags")
        .await
        .is_ok()
}

#[tokio::test]
async fn test_basic_chat() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not running");
        return;
    }

    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .max_tokens(100)
        .build();

    let reply = session.chat("Say hello in one word.").await.unwrap();
    assert!(!reply.is_empty(), "Expected non-empty reply, got empty string");
    assert_eq!(session.context().len(), 1);
}

#[tokio::test]
async fn test_multi_turn_chat() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not running");
        return;
    }

    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .max_tokens(100)
        .build();

    let _ = session.chat("My name is Alice.").await.unwrap();
    let reply = session.chat("What is my name?").await.unwrap();

    // The model should remember the name from context
    assert!(!reply.is_empty());
    assert_eq!(session.context().len(), 2);
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Color {
    name: String,
    hex: String,
}

#[tokio::test]
async fn test_structured_output() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not running");
        return;
    }

    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .max_tokens(200)
        .build();

    let color: Color = session
        .generate(
            Instruction::new("Give me a color. Pick red.")
                .response_json::<Color>()
                .require(Requirement::new("must be valid JSON").parses_as::<Color>()),
        )
        .await
        .unwrap();

    assert!(!color.name.is_empty());
    assert!(!color.hex.is_empty());
}

#[derive(Debug, Deserialize, JsonSchema)]
struct YesNo {
    answer: String,
}

#[tokio::test]
async fn test_ivr_with_validation() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not running");
        return;
    }

    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .max_tokens(100)
        .build();

    let result: YesNo = session
        .generate(
            Instruction::new("Is the sky blue? Answer yes or no.")
                .response_json::<YesNo>()
                .require(
                    Requirement::new("answer must be yes or no").validate(|s| {
                        let lower = s.to_lowercase();
                        lower.contains("yes") || lower.contains("no")
                    }),
                ),
        )
        .await
        .unwrap();

    let lower = result.answer.to_lowercase();
    assert!(
        lower.contains("yes") || lower.contains("no"),
        "Expected 'yes' or 'no', got: {}",
        result.answer
    );
}

#[tokio::test]
async fn test_session_reset() {
    if !ollama_available().await {
        eprintln!("SKIP: Ollama not running");
        return;
    }

    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .max_tokens(50)
        .build();

    let _ = session.chat("Hello").await.unwrap();
    assert_eq!(session.context().len(), 1);

    session.reset();
    assert_eq!(session.context().len(), 0);
}
