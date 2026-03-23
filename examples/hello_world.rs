//! # Hello World — Basic mellea session
//!
//! Demonstrates the simplest mellea usage: create a session, send a chat message, get a response.
//!
//! ## Running
//! ```sh
//! # Requires Ollama running on localhost:11434 with granite4:micro
//! cargo run --example hello_world
//!
//! # Or use LM Studio on localhost:1234
//! cargo run --example hello_world -- --lmstudio
//! ```

use mellea::{OllamaBackend, OpenAICompatBackend, Session};

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

    println!("Sending chat message...");
    let reply = session.chat("What is Rust? Answer in one sentence.").await?;
    println!("Response: {reply}");

    // Multi-turn: the session remembers context
    let reply2 = session.chat("What are its main advantages?").await?;
    println!("Follow-up: {reply2}");

    println!("\nConversation turns: {}", session.context().len());
    Ok(())
}
