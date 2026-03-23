# mellea-rust: Specification

**Version:** 0.1.0-poc
**Date:** 2026-03-23
**Status:** Draft

---

## 1. Why This POC Exists

### The Problem

Rust developers integrating LLMs into applications write the same fragile boilerplate
over and over: serialize a prompt, call an HTTP API, hope the response is valid JSON,
parse it, handle the inevitable malformed output, maybe retry, give up and return a
default. This code is tedious, error-prone, and scattered across every callsite.

**The result**: LLM-powered Rust code is unreliable, hard to test, and painful to maintain.

### What This POC Demonstrates

This POC proves that a thin, idiomatic Rust library can make LLM calls as **reliable and
type-safe as function calls** — with automatic validation, repair, and retry. Specifically:

1. **Structured output that actually works** — declare a Rust type, get that type back.
   The library handles JSON schema injection, parsing, and repair if the LLM fumbles.
2. **The IVR (Instruct-Validate-Repair) loop** — the core innovation from mellea-python.
   Custom validation requirements feed failure context back to the LLM, which self-corrects.
   This demonstrably improves output quality vs. raw calls.
3. **Generative functions** — declare a function signature with types, and the LLM provides
   the implementation. This is the "generative programming" paradigm.
4. **Minimal API surface** — a Rust developer goes from zero to validated LLM output in <20
   lines, using patterns they already know (builders, traits, derive macros, serde).

### What Is Mellea?

Mellea is a **generative programming library** — it replaces brittle prompt engineering and
fragile agent loops with **structured, validated, type-safe AI workflows**.

The core insight: LLM calls should behave like typed function calls. You declare what you
want (a struct, an enum variant, a typed return value), and mellea handles generation,
validation, repair, and retry — returning a value that satisfies your constraints or failing
loudly.

### Before and After: The Case for mellea

#### WITHOUT mellea — raw ollama-rs

```rust
use ollama_rs::Ollama;
use ollama_rs::generation::chat::{ChatMessage, request::ChatMessageRequest};
use serde::Deserialize;

#[derive(Deserialize)]
struct Sentiment { label: String, confidence: f64 }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ollama = Ollama::default();
    let prompt = r#"Classify the sentiment of this text as JSON.
        Output ONLY valid JSON matching: {"label": "positive"|"negative"|"neutral", "confidence": 0.0-1.0}
        Text: "I absolutely love writing Rust code!""#;

    let messages = vec![ChatMessage::user(prompt.to_string())];
    let req = ChatMessageRequest::new("granite4:micro".into(), messages);
    let res = ollama.send_chat_messages(req).await?;

    let content = res.message.content;

    // Hope it's valid JSON... often it isn't
    let sentiment: Sentiment = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(e) => {
            // LLM returned markdown-wrapped JSON? Try stripping it.
            let stripped = content.trim().trim_start_matches("```json")
                .trim_end_matches("```").trim();
            match serde_json::from_str(stripped) {
                Ok(s) => s,
                Err(_) => {
                    // Give up. Retry? With what prompt? How many times?
                    eprintln!("Failed to parse LLM output: {e}");
                    eprintln!("Raw output was: {content}");
                    return Err(e.into());
                }
            }
        }
    };

    // No validation that label is actually one of the three options
    // No validation that confidence is in range
    // No retry with feedback if the LLM gets it wrong
    // If we add validation, we need to re-prompt manually, format the error
    // message, manage attempt history, decide when to give up...

    println!("{}: {}", sentiment.label, sentiment.confidence);
    Ok(())
}
```

**Problems**: Manual JSON coaxing. No schema enforcement. No validation. No retry.
No feedback to the LLM. Every callsite repeats this. 30+ lines for a simple classification.

#### WITH mellea

```rust
use mellea::{Session, Instruction, Requirement, OllamaBackend};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Sentiment {
    /// One of: positive, negative, neutral
    label: String,
    /// Confidence score between 0.0 and 1.0
    confidence: f64,
}

#[tokio::main]
async fn main() -> Result<(), mellea::Error> {
    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .build();

    let sentiment: Sentiment = session
        .generate(
            Instruction::new("Classify the sentiment of: 'I love writing Rust!'")
                .response_json::<Sentiment>()
                .require(
                    Requirement::new("label must be positive, negative, or neutral")
                        .validate(|s| ["positive", "negative", "neutral"]
                            .contains(&s.label.as_str()))
                )
                .require(
                    Requirement::new("confidence must be between 0.0 and 1.0")
                        .validate(|s| (0.0..=1.0).contains(&s.confidence))
                ),
        )
        .await?;

    // Guaranteed: valid JSON, correct type, label is one of three options,
    // confidence is in range. If the LLM got it wrong, mellea told it why
    // and gave it another chance. If it still failed, you get a clear error.

    println!("{}: {}", sentiment.label, sentiment.confidence);
    Ok(())
}
```

**Benefits**: Type-safe. Schema-enforced. Validated. Self-repairing. 20 lines. Zero boilerplate.

#### Generative Functions — Zero-Body Typed LLM Calls

```rust
// WITHOUT mellea: you'd write the prompt, call the API, parse JSON, validate, retry...
// WITH mellea:

let extract_entities = GenerativeSlot::<Vec<Entity>>::builder()
    .name("extract_entities")
    .description("Extract named entities (people, places, organizations) from text")
    .param::<String>("text", "The text to extract entities from")
    .build();

let entities: Vec<Entity> = extract_entities
    .call(&mut session, json!({"text": "Tim Cook announced new Apple products in Cupertino"}))
    .await?;
// entities = [Entity { name: "Tim Cook", kind: Person }, Entity { name: "Apple", kind: Org }, ...]
```

### Core Value Proposition (Rust-specific)

Rust developers working with LLMs today face a choice between:

| Approach | Problem |
|---|---|
| Raw HTTP / ollama-rs | No structure, no validation, string-in string-out |
| rig (agent framework) | Full agent abstraction, but no structured output validation or retry loops |
| genai (multi-provider client) | Thin HTTP normalization, no generation logic |
| rstructor (structured output) | Basic retry, limited providers, low adoption (~64 downloads/month) |

**mellea fills the gap**: type-safe generative programming with IVR (Instruct-Validate-Repair)
loops, built on top of established crates, idiomatic to Rust.

**A Rust developer using mellea gets better, more reliable LLM results than without it** —
because every generation is validated against type constraints and custom requirements, with
automatic repair feedback when validation fails.

---

## 2. Competitive Landscape

### What exists in Rust (2026-03)

| Crate | Stars | Downloads/mo | Focus | Gap |
|---|---|---|---|---|
| **rig** (0xPlaygrounds) | 6,594 | High | Agent framework, 20+ providers, tool calling | No structured output validation, no IVR |
| **genai** | 692 | ~15k | Multi-provider chat API normalization | Thin client, no generation logic |
| **ollama-rs** | ~500 | 25k | Ollama client | Raw client, no orchestration |
| **swiftide** | ~300 | ~21k | RAG/indexing pipelines | Pipeline focus, not generative programming |
| **kalosm** | ~300 | ~6k | Local inference with constrained decoding | Local-only, candle-based |
| **rstructor** | ~50 | ~64 | Instructor-like structured output | Low adoption, limited providers |
| **llm-chain** | ~400 | Dead | LangChain port | Abandoned since 2023 |

### What's missing

1. **Generative functions**: No Rust crate offers `#[generative] fn classify(text: &str) -> Sentiment`
2. **IVR loops**: No crate offers configurable instruct-validate-repair with error feedback
3. **Composable validation**: No crate lets you attach custom `Requirement` validators that feed
   failure context back to the LLM for repair
4. **Provider-agnostic structured generation**: rstructor is the closest but has negligible adoption

### Python equivalents

| Python | Rust equivalent | mellea-rust target |
|---|---|---|
| mellea (Python) | Nothing | Direct port of core concepts |
| Instructor | rstructor (weak) | IVR + structured output |
| Pydantic AI | Nothing | Generative functions |
| BAML | Nothing | Type-safe LLM function signatures |

---

## 3. Architecture

### Crate Structure

```
mellea-rust/
├── Cargo.toml              # workspace root
├── mellea/                  # core library crate
│   ├── src/
│   │   ├── lib.rs
│   │   ├── backend/        # Backend trait + implementations
│   │   │   ├── mod.rs
│   │   │   ├── ollama.rs   # Ollama via ollama-rs
│   │   │   └── openai_compat.rs  # OpenAI-compatible (LM Studio, etc.)
│   │   ├── core/           # Fundamental types
│   │   │   ├── mod.rs
│   │   │   ├── context.rs  # Conversation context (immutable linked list)
│   │   │   ├── component.rs # Component trait
│   │   │   ├── block.rs    # CBlock, ModelOutput
│   │   │   └── thunk.rs    # ModelOutputThunk<T> — lazy typed output
│   │   ├── session/        # MelleaSession
│   │   │   ├── mod.rs
│   │   │   └── builder.rs  # Session builder
│   │   ├── ivr/            # Instruct-Validate-Repair loop
│   │   │   ├── mod.rs
│   │   │   ├── requirement.rs  # Requirement + ValidationResult
│   │   │   ├── strategy.rs     # SamplingStrategy trait
│   │   │   └── rejection.rs    # Rejection sampling (default)
│   │   ├── component/      # Built-in components
│   │   │   ├── mod.rs
│   │   │   ├── instruction.rs  # Instruction component
│   │   │   └── message.rs      # Chat Message component
│   │   ├── generative/     # Generative function support
│   │   │   ├── mod.rs
│   │   │   └── slot.rs     # GenerativeSlot builder
│   │   └── error.rs        # Error types
│   └── Cargo.toml
├── mellea-macros/           # proc-macro crate (future: #[generative])
│   ├── src/lib.rs
│   └── Cargo.toml
├── examples/                # Sample applications
│   ├── hello_world.rs
│   ├── sentiment.rs
│   ├── ivr_demo.rs
│   └── generative_fn.rs
├── SPEC.md
└── README.md
```

### Dependency Strategy

**Principle**: Use well-established, actively maintained crates with real adoption.

| Dependency | Purpose | Stars/Downloads | Justification |
|---|---|---|---|
| `tokio` | Async runtime | 28k stars | Universal standard |
| `serde` / `serde_json` | Serialization | 9k stars | Universal standard |
| `schemars` | JSON Schema from Rust types | 1k stars | Used by ollama-rs, widely adopted |
| `ollama-rs` | Ollama client | ~500 stars, 25k dl/mo | De facto Ollama client for Rust |
| `reqwest` | HTTP client (OpenAI-compat) | 10k stars | Universal standard |
| `thiserror` | Error types | 5k stars | Universal standard |
| `tracing` | Observability | 6k stars | Universal standard |
| `async-trait` | Async trait support | 2k stars | Until RPITIT stabilizes fully |

**No obscure or low-adoption dependencies.** Every crate listed above has >500 stars and
significant real-world usage.

---

## 3.1 Python to Rust: Design Translation

mellea-rust is not a line-by-line Python port. It translates mellea's concepts into
idiomatic Rust patterns, leveraging the type system, ownership model, and ecosystem.

| Python (mellea) | Rust (mellea-rust) | Rationale |
|---|---|---|
| `MelleaSession` class + `__init__` args | `Session` struct + `SessionBuilder` | Rust builder pattern for complex configuration; no default mutable args |
| Pydantic `BaseModel` for schemas | `#[derive(Deserialize, JsonSchema)]` | serde + schemars = Pydantic for Rust; compile-time schema generation |
| `@generative` decorator (runtime reflection) | `GenerativeSlot::<T>::builder()` | No runtime reflection in Rust; explicit type-parameterized builders |
| `Requirement(validation_fn=lambda ctx: ...)` | `Requirement::new("desc").validate(\|s\| ...)` | Closures as validators, fluent builder chain |
| `MelleaSession.act()` returns `(output, ctx)` | `Session` owns mutable `Context` | Rust ownership replaces manual context threading |
| `SimpleContext()` immutable linked list | `Context { turns: Vec<Turn> }` with `.push()` returning new `Context` | `Vec` is natural in Rust; immutability via clone-on-write |
| `ModelOutputThunk[T]` lazy evaluation | `ModelOutput.parse::<T>()` explicit parse | No lazy magic; explicit type-directed parsing via turbofish |
| `try/except` with repair prompt | `Result<T, MelleaError>` + IVR loop | Rust's type-safe error handling; `MelleaError` enum is matchable |
| Multiple backends via class hierarchy | `Backend` trait + `dyn Backend` | Trait objects for runtime polymorphism |
| Plugin hooks (`GENERATION_PRE_CALL`, etc.) | Not in POC | Minimal surface area for POC |
| `model_options: dict` | `ModelOptions` struct with typed fields | Compile-time validation of options |
| Dynamic typing for LLM responses | Generics: `generate::<T>()` | Type parameter determines parse target at compile time |

### Minimum Rust Version

Requires **Rust 1.94+** (edition 2024). No backwards-compatibility shims.
Uses current language features (edition 2024 reserved keywords, modern trait syntax).

---

## 4. Core Abstractions

### 4.1 Backend Trait

The abstraction over LLM providers. Implementations provided for Ollama (via ollama-rs) and
OpenAI-compatible APIs (covering LM Studio, vLLM, etc.).

```rust
#[async_trait]
pub trait Backend: Send + Sync {
    /// Generate a completion from conversation context.
    async fn generate(&self, request: GenerateRequest) -> Result<ModelOutput, MelleaError>;

    /// Generate with streaming callback.
    async fn generate_stream(
        &self,
        request: GenerateRequest,
        on_token: impl Fn(&str) + Send + 'static,
    ) -> Result<ModelOutput, MelleaError>;
}

pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub options: ModelOptions,
    pub tools: Option<Vec<ToolDefinition>>,
    pub response_format: Option<ResponseFormat>,
}

pub struct ModelOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<Vec<String>>,
}
```

### 4.2 Context

Immutable conversation history. Each operation returns a new context — no mutation.

```rust
pub struct Context {
    turns: Vec<Turn>,
}

pub struct Turn {
    pub input: ChatMessage,
    pub output: Option<ModelOutput>,
}

impl Context {
    pub fn new() -> Self;
    pub fn push(&self, turn: Turn) -> Self;  // returns new Context
    pub fn last_output(&self) -> Option<&ModelOutput>;
    pub fn messages(&self) -> Vec<ChatMessage>;
    pub fn len(&self) -> usize;
}
```

### 4.3 ModelOutput

The result of an LLM generation, with typed parsing.

```rust
pub struct ModelOutput {
    pub content: String,
    pub model: String,
    pub usage: Usage,
    pub tool_calls: Vec<ToolCall>,
    pub raw: serde_json::Value,
}

pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl ModelOutput {
    /// Parse the content as a typed value.
    /// Uses serde_json for JSON responses, with schema-guided parsing.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T, ParseError>;
}
```

### 4.4 Requirement + ValidationResult

The validation layer. Requirements can be schema-based (automatic from serde), custom
functions, or LLM-as-judge.

```rust
pub struct Requirement {
    /// Human-readable description (included in repair prompts).
    pub description: String,
    /// The validation function.
    pub validate: Box<dyn Fn(&str, &Context) -> ValidationResult + Send + Sync>,
}

pub enum ValidationResult {
    Pass,
    Fail { reason: String },
}

// Builder for ergonomic requirement construction
impl Requirement {
    pub fn new(description: impl Into<String>) -> RequirementBuilder;
}

pub struct RequirementBuilder { ... }

impl RequirementBuilder {
    /// Validate with a closure.
    pub fn validate(self, f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Requirement;

    /// Validate with a closure that returns a reason on failure.
    pub fn validate_with_reason(
        self,
        f: impl Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    ) -> Requirement;

    /// Validate that output parses as type T.
    pub fn parses_as<T: DeserializeOwned + 'static>(self) -> Requirement;

    /// Validate output length.
    pub fn max_length(self, n: usize) -> Requirement;
    pub fn min_length(self, n: usize) -> Requirement;
}
```

### 4.5 IVR Loop (Instruct-Validate-Repair)

The core generation loop. Configurable retry strategies.

```rust
#[async_trait]
pub trait SamplingStrategy: Send + Sync {
    async fn sample(
        &self,
        backend: &dyn Backend,
        request: GenerateRequest,
        requirements: &[Requirement],
        context: &Context,
    ) -> Result<SamplingResult, MelleaError>;
}

pub struct SamplingResult {
    pub output: ModelOutput,
    pub attempts: Vec<SamplingAttempt>,
    pub success: bool,
}

pub struct SamplingAttempt {
    pub output: ModelOutput,
    pub validations: Vec<(String, ValidationResult)>,
}

/// Default strategy: retry with validation feedback.
pub struct RejectionSampling {
    pub max_retries: u32,  // default: 2
}
```

**IVR flow**:
1. **Instruct**: Format the component into messages with requirements descriptions
2. **Generate**: Call backend
3. **Validate**: Check all requirements against output
4. **If any fail**: Append repair message with failure reasons, go to step 2
5. **If all pass or budget exhausted**: Return SamplingResult

### 4.6 Session

The primary user-facing API. Manages context threading and backend configuration.

```rust
pub struct Session {
    backend: Arc<dyn Backend>,
    context: Context,
    default_options: ModelOptions,
    default_strategy: Arc<dyn SamplingStrategy>,
}

impl Session {
    pub fn builder() -> SessionBuilder;

    /// Simple chat — send a message, get a response.
    pub async fn chat(&mut self, content: &str) -> Result<String, MelleaError>;

    /// Instruct with requirements and IVR.
    pub async fn instruct(
        &mut self,
        instruction: Instruction,
    ) -> Result<ModelOutput, MelleaError>;

    /// Generate a typed value via structured output + IVR.
    pub async fn generate<T>(&mut self, instruction: Instruction) -> Result<T, MelleaError>
    where
        T: DeserializeOwned + JsonSchema;

    /// Access the conversation context.
    pub fn context(&self) -> &Context;

    /// Reset context.
    pub fn reset(&mut self);
}
```

### 4.7 Instruction Component

The primary way to specify what you want from the LLM.

```rust
pub struct Instruction {
    pub description: String,
    pub requirements: Vec<Requirement>,
    pub grounding: HashMap<String, String>,
    pub examples: Vec<Example>,
    pub response_format: Option<ResponseFormat>,
}

pub struct Example {
    pub input: String,
    pub output: String,
}

pub enum ResponseFormat {
    Json,
    JsonSchema(schemars::schema::RootSchema),
    Text,
}

impl Instruction {
    pub fn new(description: impl Into<String>) -> Self;

    pub fn require(mut self, req: Requirement) -> Self;
    pub fn ground(mut self, key: impl Into<String>, value: impl Into<String>) -> Self;
    pub fn example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self;
    pub fn response_json<T: JsonSchema>(mut self) -> Self;
}
```

### 4.8 Generative Functions (POC)

For the POC, generative functions use a builder pattern rather than proc macros.
The proc-macro crate (`mellea-macros`) is scaffolded but the `#[generative]` attribute
is a stretch goal.

```rust
/// Define a generative function via builder.
let classify = GenerativeSlot::<Sentiment>::builder()
    .name("classify_sentiment")
    .description("Classify the sentiment of the given text")
    .param::<String>("text", "The text to classify")
    .require(Requirement::new("must be a valid Sentiment variant")
        .parses_as::<Sentiment>())
    .build();

/// Call it
let result: Sentiment = classify
    .call(&mut session, serde_json::json!({"text": "I love Rust!"}))
    .await?;

// Stretch goal: proc macro
#[generative]
async fn classify_sentiment(session: &mut Session, text: String) -> Result<Sentiment, MelleaError> {
    // no body — LLM fills this in
}
```

---

## 5. Backend Implementations

### 5.1 Ollama Backend (via ollama-rs)

Primary backend. Uses the well-established `ollama-rs` crate.

```rust
pub struct OllamaBackend {
    client: ollama_rs::Ollama,
    model: String,
}

impl OllamaBackend {
    pub fn new(model: impl Into<String>) -> Self;
    pub fn with_host(host: impl Into<String>, port: u16, model: impl Into<String>) -> Self;
}
```

Default: `localhost:11434`, model `granite4:micro`.

### 5.2 OpenAI-Compatible Backend (for LM Studio, vLLM, etc.)

Uses `reqwest` directly against the OpenAI chat completions API.

```rust
pub struct OpenAICompatBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl OpenAICompatBackend {
    /// LM Studio default: localhost:1234
    pub fn lmstudio(model: impl Into<String>) -> Self;

    /// Custom endpoint
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self;

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self;
}
```

---

## 6. POC Scope

### Must Have (MVP)

- [ ] Core types: `Context`, `ModelOutput`, `ChatMessage`, `Usage`
- [ ] `Backend` trait + `OllamaBackend` implementation
- [ ] `OpenAICompatBackend` (LM Studio support)
- [ ] `Session` with builder pattern, `chat()`, `instruct()`, `generate::<T>()`
- [ ] `Instruction` component with builder pattern
- [ ] `Requirement` + `ValidationResult` with builder helpers
- [ ] IVR loop: `RejectionSampling` strategy
- [ ] `GenerativeSlot` builder for generative functions
- [ ] JSON structured output via `schemars` + `serde`
- [ ] Error types via `thiserror`
- [ ] Async throughout via `tokio`
- [ ] Example: hello world (basic chat)
- [ ] Example: sentiment classifier (generative function)
- [ ] Example: IVR demo (instruction with validation + repair)
- [ ] Example: generative function with custom requirements
- [ ] Unit tests for core types
- [ ] Integration tests (require running Ollama)
- [ ] README positioning the POC

### Nice to Have (Stretch)

- [ ] `mellea-macros` proc-macro crate with `#[generative]` attribute
- [ ] Streaming support
- [ ] Tool calling
- [ ] `tracing` instrumentation
- [ ] Multiple sampling strategies (majority voting, best-of-n)

### Explicitly Out of Scope

- Plugin/hook system
- RAG / vector store integration
- Multiple LLM provider SDKs (beyond Ollama + OpenAI-compat)
- Web server / HTTP endpoint serving
- Vision / multimodal
- Prompt templates / Jinja equivalent

---

## 7. Design Principles

### 7.1 Idiomatic Rust

- **Builder pattern** for all configuration (Session, Instruction, Requirement, Backend)
- **Trait-based extension** for Backend, SamplingStrategy
- **Immutable by default** — Context is append-only, operations return new values
- **Strong typing** — `schemars::JsonSchema` + `serde::Deserialize` for structured output.
  The LLM must return valid typed data or the IVR loop repairs it.
- **Error handling** — `thiserror` enums, no panics, no silent failures
- **Async-first** — `tokio` runtime, `async fn` everywhere

### 7.2 Minimal Surface Area

The POC exposes a small, composable API:

```rust
use mellea::{Session, Instruction, Requirement, OllamaBackend};

#[tokio::main]
async fn main() -> Result<(), mellea::Error> {
    let mut session = Session::builder()
        .backend(OllamaBackend::new("granite4:micro"))
        .build();

    // Simple chat
    let reply = session.chat("What is Rust?").await?;
    println!("{reply}");

    // Structured output with validation
    let sentiment: Sentiment = session
        .generate(
            Instruction::new("Classify the sentiment of: 'I love Rust!'")
                .response_json::<Sentiment>()
                .require(Requirement::new("valid sentiment").parses_as::<Sentiment>()),
        )
        .await?;

    Ok(())
}
```

### 7.3 Build on Established Crates

No reinventing HTTP clients, serialization, or async runtimes. mellea's value is in the
**generation logic layer** — the IVR loop, validation, structured output, and generative
function abstraction. Everything below that uses battle-tested crates.

### 7.4 Fail Fast and Loud

- Validation failures produce detailed `MelleaError::ValidationFailed` with all attempt
  history, failure reasons, and raw LLM outputs
- No silent retries beyond the configured budget
- No `unwrap()` in library code
- All errors are structured and inspectable

---

## 8. Model Defaults

| Setting | Default |
|---|---|
| Backend | Ollama (`localhost:11434`) |
| Model | `granite4:micro` |
| Max tokens | 1024 |
| Temperature | 0.7 |
| IVR max retries | 2 |
| Response format | Text (unless `response_json::<T>()` specified) |

Both Ollama and LM Studio are confirmed running locally with Granite 4 models available.

---

## 9. Testing Strategy

### Unit Tests
- Context operations (push, messages, last_output)
- Requirement validation (parses_as, max_length, custom)
- Instruction building
- Message formatting
- Error types

### Integration Tests (require running Ollama)
- Basic chat roundtrip
- Structured output generation + parsing
- IVR loop: intentionally trigger validation failure, verify repair attempt
- GenerativeSlot invocation
- OpenAI-compat backend against LM Studio

### Example Programs
Each example is a standalone `main()` that demonstrates a core feature:
1. `hello_world.rs` — Session + chat
2. `sentiment.rs` — Generative function (typed output)
3. `ivr_demo.rs` — IVR loop with custom requirements
4. `generative_fn.rs` — GenerativeSlot builder pattern

---

## 10. Success Criteria

The POC is successful if:

1. A Rust developer can `cargo add mellea` and get validated, typed LLM responses in <20 lines
2. The IVR loop demonstrably improves output quality vs. raw LLM calls
3. Generative functions feel natural to Rust developers (builder or attribute)
4. Code compiles with zero warnings, all tests pass
5. Examples run against local Ollama and produce correct results
