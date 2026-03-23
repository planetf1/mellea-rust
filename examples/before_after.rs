//! # Before & After — The case for mellea
//!
//! This example runs the **same task** two ways:
//!
//! 1. **WITHOUT mellea** — raw ollama-rs, manual JSON parsing, no validation, no retry
//! 2. **WITH mellea** — structured output, IVR validation, automatic repair
//!
//! The "before" code may fail or return garbage. The "after" code is constrained
//! and self-repairing. Run it yourself to see the difference.
//!
//! ## Running
//! ```sh
//! cargo run --example before_after
//! ```

use schemars::JsonSchema;
use serde::Deserialize;

// ─── The task: extract a structured movie review from free text ───

const REVIEW_TEXT: &str = r#"
Just watched "The Matrix" again last night. Still holds up after all these years!
The special effects were groundbreaking and Keanu's performance is iconic.
I'd give it a solid 9 out of 10. Definitely a must-watch for any sci-fi fan.
"#;

/// The structured output we want from both approaches.
#[derive(Debug, Deserialize, JsonSchema)]
struct MovieReview {
    /// The movie title
    title: String,
    /// Rating from 1-10
    rating: u8,
    /// One of: positive, negative, mixed
    sentiment: String,
    /// One-sentence summary
    summary: String,
}

// ═══════════════════════════════════════════════════════════════════
// BEFORE: Raw ollama-rs — no mellea
// ═══════════════════════════════════════════════════════════════════

async fn without_mellea() {
    println!("═══ WITHOUT mellea (raw ollama-rs) ═══\n");

    let ollama = ollama_rs::Ollama::default();

    let prompt = format!(
        r#"Extract a movie review from this text as JSON.
Output ONLY valid JSON with fields: title, rating (1-10), sentiment (positive/negative/mixed), summary.
Text: {REVIEW_TEXT}"#
    );

    let messages = vec![ollama_rs::generation::chat::ChatMessage::user(prompt)];
    let mut req =
        ollama_rs::generation::chat::request::ChatMessageRequest::new("granite4:micro".into(), messages);
    req = req.format(ollama_rs::generation::parameters::FormatType::Json);

    match ollama.send_chat_messages(req).await {
        Ok(response) => {
            let raw = &response.message.content;
            println!("Raw LLM output:\n{raw}\n");

            // Try to parse it...
            match serde_json::from_str::<MovieReview>(raw) {
                Ok(review) => {
                    println!("Parsed successfully!");
                    println!("  Title:     {}", review.title);
                    println!("  Rating:    {}", review.rating);
                    println!("  Sentiment: {}", review.sentiment);
                    println!("  Summary:   {}", review.summary);

                    // But is the data actually valid?
                    let mut issues = Vec::new();
                    if review.rating == 0 || review.rating > 10 {
                        issues.push(format!("rating {} is outside 1-10", review.rating));
                    }
                    if !["positive", "negative", "mixed"].contains(&review.sentiment.as_str()) {
                        issues.push(format!("sentiment '{}' is not positive/negative/mixed", review.sentiment));
                    }
                    if review.summary.len() > 200 {
                        issues.push(format!("summary is {} chars (too long)", review.summary.len()));
                    }
                    if review.title.is_empty() {
                        issues.push("title is empty".to_string());
                    }

                    if issues.is_empty() {
                        println!("  Quality:   PASS (got lucky this time)\n");
                    } else {
                        println!("  Quality:   FAIL");
                        for issue in &issues {
                            println!("    - {issue}");
                        }
                        println!("  (No way to fix this without re-prompting manually)\n");
                    }
                }
                Err(e) => {
                    println!("PARSE FAILED: {e}");
                    println!("  The LLM returned something that isn't valid MovieReview JSON.");
                    println!("  You'd need to: strip markdown fences, retry, build repair prompts...");
                    println!("  All of that is manual boilerplate.\n");
                }
            }
        }
        Err(e) => {
            println!("LLM CALL FAILED: {e}");
            println!("  No retry logic, no fallback, no context.\n");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// AFTER: With mellea — structured, validated, self-repairing
// ═══════════════════════════════════════════════════════════════════

async fn with_mellea() -> Result<(), mellea::Error> {
    println!("═══ WITH mellea (structured + validated) ═══\n");

    let mut session = mellea::Session::builder()
        .backend(mellea::OllamaBackend::new("granite4:micro"))
        .build();

    let review: MovieReview = session
        .generate(
            mellea::Instruction::new("Extract a movie review from this text.")
                .ground("text", REVIEW_TEXT)
                .response_json::<MovieReview>()
                .require(
                    mellea::Requirement::new("rating must be between 1 and 10")
                        .validate_with_reason(|s| {
                            let v: serde_json::Value = serde_json::from_str(s)
                                .map_err(|e| format!("not valid JSON: {e}"))?;
                            let r = v.get("rating").and_then(|r| r.as_u64())
                                .ok_or("missing or non-numeric 'rating' field")?;
                            if (1..=10).contains(&r) {
                                Ok(())
                            } else {
                                Err(format!("rating {r} is outside 1-10 range"))
                            }
                        }),
                )
                .require(
                    mellea::Requirement::new("sentiment must be positive, negative, or mixed")
                        .validate_with_reason(|s| {
                            let v: serde_json::Value = serde_json::from_str(s)
                                .map_err(|e| format!("not valid JSON: {e}"))?;
                            let sent = v.get("sentiment").and_then(|s| s.as_str())
                                .ok_or("missing 'sentiment' field")?;
                            if ["positive", "negative", "mixed"].contains(&sent) {
                                Ok(())
                            } else {
                                Err(format!("sentiment '{sent}' must be positive, negative, or mixed"))
                            }
                        }),
                )
                .require(
                    mellea::Requirement::new("summary must be under 200 characters")
                        .validate_with_reason(|s| {
                            let v: serde_json::Value = serde_json::from_str(s)
                                .map_err(|e| format!("not valid JSON: {e}"))?;
                            let summary = v.get("summary").and_then(|s| s.as_str())
                                .ok_or("missing 'summary' field")?;
                            if summary.len() <= 200 {
                                Ok(())
                            } else {
                                Err(format!("summary is {} chars, must be <= 200", summary.len()))
                            }
                        }),
                ),
        )
        .await?;

    // If we get here, ALL requirements passed. The data is guaranteed valid.
    println!("Result (guaranteed valid):");
    println!("  Title:     {}", review.title);
    println!("  Rating:    {} (validated: 1-10)", review.rating);
    println!("  Sentiment: {} (validated: positive/negative/mixed)", review.sentiment);
    println!("  Summary:   {} ({} chars, validated: <200)", review.summary, review.summary.len());
    println!("  Quality:   GUARANTEED PASS\n");

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    println!("╔════════════════════════════════════════════════════╗");
    println!("║  mellea Before & After: Same task, two approaches ║");
    println!("╚════════════════════════════════════════════════════╝\n");

    // Run WITHOUT mellea first
    without_mellea().await;

    // Run WITH mellea
    match with_mellea().await {
        Ok(()) => {}
        Err(e) => {
            println!("mellea error (after retries exhausted): {e}");
            println!("  Even failure is structured — you get the full attempt history.");
        }
    }

    println!("─────────────────────────────────────────────────────");
    println!("Key difference: the 'before' code may silently return bad data.");
    println!("The 'after' code guarantees valid, constrained output — or fails loudly.");
}
