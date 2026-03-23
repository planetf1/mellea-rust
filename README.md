# mellea-rust

**Type-safe generative programming for Rust** — structured LLM output with
instruct-validate-repair loops.

mellea replaces brittle prompt engineering with validated, typed AI workflows.
Declare what you want as a Rust struct, and mellea handles generation, validation,
repair, and retry — returning a value that satisfies your constraints or failing loudly.

> This is a **proof-of-concept** port of [mellea](https://github.com/generative-computing/mellea)
> (Python) to Rust. It demonstrates that the core generative programming paradigm
> translates naturally to Rust's type system and ownership model.

## Why mellea?

Rust developers working with LLMs today write the same fragile boilerplate everywhere:
build a prompt, call an API, hope for valid JSON, parse it, handle malformed output,
maybe retry, give up. Every callsite repeats this. The output is unreliable.

**mellea makes LLM calls as reliable as function calls:**

| Without mellea | With mellea |
|---|---|
| Manual JSON prompt injection | `response_json::<T>()` — schema generated from types |
| Hand-written parse/retry loops | IVR loop with automatic repair feedback |
| `serde_json::from_str` and pray | Type-safe `session.generate::<T>()` |
| 30+ lines per structured call | ~5 lines per structured call |
| No validation of LLM output | Custom `Requirement` validators with repair context |

## Quick Start

```rust
use mellea::{Session, Instruction, Requirement, OllamaBackend};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Sentiment {
    label: String,
    confidence: f64,
}

#[tokio::main]
async fn main() -> Result<(), mellea::Error> {
    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .build();

    let sentiment: Sentiment = session
        .generate(
            Instruction::new("Classify the sentiment of: 'I love Rust!'")
                .response_json::<Sentiment>()
                .require(
                    Requirement::new("label must be positive, negative, or neutral")
                        .validate(|s| {
                            s.contains("positive") || s.contains("negative") || s.contains("neutral")
                        }),
                ),
        )
        .await?;

    println!("{}: {:.2}", sentiment.label, sentiment.confidence);
    Ok(())
}
```

## Core Concepts

### IVR Loop (Instruct-Validate-Repair)

The core innovation: when the LLM returns invalid output, mellea tells it
*why* it failed and asks it to fix it. This self-repair loop demonstrably
improves output quality vs. raw LLM calls.

```
Instruction → Generate → Validate → [Pass] → Return typed result
                 ↑          ↓
                 └── Repair feedback ←── [Fail: "label was 'good', must be positive/negative/neutral"]
```

### Generative Functions

Declare a function signature with types. The LLM provides the implementation.

```rust
let extract = GenerativeSlot::<Entities>::builder()
    .name("extract_entities")
    .description("Extract named entities from text")
    .param::<String>("text", "Text to process")
    .build();

let entities: Entities = extract
    .call(&mut session, json!({"text": "Tim Cook at Apple in Cupertino"}))
    .await?;
```

### Backends

- **Ollama** (default) — via `ollama-rs`, connects to `localhost:11434`
- **OpenAI-compatible** — LM Studio, vLLM, llama.cpp, any `/v1/chat/completions` endpoint

```rust
// LM Studio
Session::builder().backend(OpenAICompatBackend::lmstudio("granite-4.0-h-micro")).build();

// Custom endpoint
Session::builder().backend(OpenAICompatBackend::new("http://myserver:8080/v1", "model")).build();
```

## Python to Rust: Design Translation

mellea-rust is not a line-by-line port. It translates mellea-python's concepts
into idiomatic Rust patterns:

| Python (mellea) | Rust (mellea-rust) | Why |
|---|---|---|
| `MelleaSession` class | `Session` struct + builder | Rust builder pattern for configuration |
| Pydantic `BaseModel` | `#[derive(Deserialize, JsonSchema)]` | serde + schemars = Pydantic for Rust |
| `@generative` decorator | `GenerativeSlot::<T>::builder()` | No runtime reflection; explicit builders |
| `Requirement(validation_fn=lambda)` | `Requirement::new("desc").validate(\|s\| ...)` | Closures as validators, builder chain |
| `MelleaSession.act()` context threading | `Session` owns mutable `Context` | Ownership replaces manual ctx passing |
| `SimpleContext()` linked list | `Context { turns: Vec<Turn> }` | Vec is natural in Rust |
| `ModelOutputThunk[T]` lazy eval | `ModelOutput.parse::<T>()` | Explicit parse, no lazy magic |
| `try/except` with repair prompt | `Result<T, MelleaError>` + IVR loop | Type-safe error handling |
| Multiple backends via class hierarchy | `Backend` trait + `dyn Backend` | Trait objects for polymorphism |
| Plugin hooks | Not in POC | Keep surface area minimal |

## Requirements

- **Rust 1.94+** (edition 2024)
- **Ollama** running on `localhost:11434` (or LM Studio on `localhost:1234`)
- A model pulled — examples default to `granite4:micro`

## Running Examples

```sh
# Basic chat
cargo run --example hello_world

# Structured output (sentiment classification)
cargo run --example sentiment

# IVR demo (validation + repair loop)
cargo run --example ivr_demo

# Generative functions
cargo run --example generative_fn

# Use LM Studio instead of Ollama
cargo run --example sentiment -- --lmstudio
```

## Running Tests

```sh
# Unit tests (no LLM required)
cargo test

# Generate documentation
cargo doc --open
```

## Project Structure

```
mellea-rust/
├── mellea/                  # Core library crate
│   └── src/
│       ├── lib.rs           # Public API, re-exports
│       ├── error.rs         # MelleaError types
│       ├── core/            # Context, ChatMessage, ModelOutput
│       ├── backend/         # Backend trait, Ollama, OpenAI-compat
│       ├── ivr/             # Requirement, ValidationResult, RejectionSampling
│       ├── session/         # Session + builder
│       ├── component/       # Instruction, Message
│       └── generative/      # GenerativeSlot builder
├── mellea-macros/           # Proc-macro crate (future: #[generative])
├── examples/                # Runnable examples
├── SPEC.md                  # Detailed specification
└── README.md
```

## Dependencies

All dependencies are well-established, actively maintained crates:

| Crate | Purpose | Justification |
|---|---|---|
| `tokio` | Async runtime | Universal Rust async standard |
| `serde` / `serde_json` | Serialization | Universal Rust serialization |
| `schemars` | JSON Schema from types | Used by ollama-rs and widely |
| `ollama-rs` | Ollama client | De facto Rust Ollama client |
| `reqwest` | HTTP client | Universal Rust HTTP |
| `thiserror` | Error types | Standard derive-based errors |
| `tracing` | Observability | Standard Rust instrumentation |
| `async-trait` | Async trait support | Widely used for dyn-safe async |

## Documentation

- [SPEC.md](SPEC.md) — Architecture, competitive analysis, design decisions
- [RETROSPECTIVE.md](RETROSPECTIVE.md) — POC process review, open questions, and next steps

## License

Apache-2.0 — same as mellea-python.
