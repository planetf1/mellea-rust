# mellea-rust POC: Retrospective

**Date:** 2026-03-23
**Duration:** Single session (~45 minutes wall clock)
**Output:** 3,452 lines, 32 files, 22 tests, 5 examples, all passing

---

## What Was Easy

### Crate scaffolding and structure
Cargo workspaces, module layout, dependency selection were mechanical. The Rust
ecosystem for LLM work (ollama-rs, serde, schemars, tokio) is mature enough that
wiring things together required no invention — just assembly.

### Core type design
Context, ModelOutput, ChatMessage, Backend trait — these translated almost 1:1 from
Python concepts to Rust idioms. Rust's type system made the design *clearer*, not
harder. Ownership semantics naturally enforced immutability on Context without needing
the discipline that Python requires.

### The IVR loop
Conceptually simple: generate, check requirements, append failure context, retry.
The implementation was ~100 lines of straightforward async code and worked correctly
on its first compile.

### Unit testing
Pure logic (JSON artifact stripping, requirement validation, context operations)
was trivial to test — 16 tests, all deterministic, zero external dependencies, run
in <1ms total.

### Competitive research
The Rust LLM ecosystem is well-indexed. Identifying the gap (no IVR, no generative
functions, fragmented structured output) was straightforward from crate surveys and
GitHub analysis.

---

## What Was Hard

### Small model prompt engineering (~40% of total effort)

The single biggest time sink. `granite4:micro` (3B parameters) repeatedly:

- **Echoed back the JSON schema** instead of filling it in. Fixed by generating compact
  example objects rather than injecting full JSON Schema documents.
- **Used wrong field names** (`text`/`label` instead of `name`/`kind`). Fixed by making
  the schema-to-example builder recurse into nested objects and array items.
- **Dropped required fields** (e.g., `confidence`). Fixed by auto-adding `parses_as::<T>()`
  in `session.generate::<T>()` so the IVR loop catches and repairs parse failures.

Each issue required: run example → observe failure → diagnose root cause → fix prompt
formatting → rerun. This was iterative and slow.

**Irony:** This is precisely the problem mellea exists to solve. The POC itself
demonstrated the value proposition during its own development.

### ollama-rs API discovery

The crate reorganized its module structure between versions. `GenerationOptions`
became `ModelOptions` and moved from `generation::options` to `models`. `Ollama::new()`
changed its signature to take `impl IntoUrl` instead of `impl Into<String>`. The only
way to discover this was reading source files in `~/.cargo/registry/`. Published docs
lagged behind the released code.

### Edition 2024 keyword collision

`gen` is a reserved keyword in Rust edition 2024. Using it as a variable name produced
a cryptic syntax error with no helpful diagnostic. Minor but an unexpected 5-minute
detour.

### Incremental feedback integration

The session received ~6 feedback messages while implementation was in-flight (add idiom
table, add before/after examples, document Rust version, add docs, ensure tests at all
levels, proper error reporting). Each was individually good but caused context switches
mid-implementation. Batching these into a spec review gate before coding would have
been smoother.

---

## What I'd Tell the Requester

### 1. The POC proves the thesis

The IVR loop and typed generation work well in Rust. The before/after example is
compelling — same task, one approach is fragile and may silently return bad data,
the other guarantees valid constrained output. The value proposition is real and
demonstrable.

### 2. The hard part isn't the library — it's the prompt layer

Getting small models to consistently produce correct JSON with exact field names
required iterating on how schemas are presented. A production version needs more
sophisticated prompt formatting — possibly model-specific strategies, or leveraging
structured output modes where available (Ollama's `StructuredJson` format type).

### 3. Scope was right for a POC

Cutting plugins, streaming, tool calling, and proc macros kept the surface area
small and focused. The `GenerativeSlot` builder works but `#[generative]` proc
macro would be the killer ergonomic feature for a real release.

### 4. The competitive landscape is favorable

No Rust crate does IVR. `rig` is the ecosystem leader but has zero validation/retry.
`rstructor` is the closest concept but has negligible adoption (~64 downloads/month).
There is clear whitespace for mellea-rust.

### 5. Decide the audience before going further

Is mellea-rust for:

| Audience | Implies |
|---|---|
| Rust developers wanting better LLM integration | Focus on ergonomics, docs, examples, crates.io presence |
| The mellea ecosystem wanting a Rust target | Focus on API compatibility with mellea-python |
| Production systems needing validated LLM output | Focus on reliability, observability, error reporting, performance |

The answer shapes API design, feature priority, and documentation strategy.

---

## What Would Make the Process Better Next Time

### 1. Pin the LLM model upfront

I defaulted to `granite4:micro` because it was available locally. A larger model
(8B+) would have eliminated most of the prompt engineering pain. Alternatively,
decide early: "POC targets model X at size Y" and tune for that.

### 2. Write the before/after example first

The before/after comparison was the most valuable artifact. It should be the
*starting point* — "here's the demo we want to show" — with everything else
built to support it. Design from the demo backwards.

### 3. Gate spec review before coding

The spec was written and coding started immediately. A pause to review the spec
(with the requester) would have surfaced the before/after requirement, the idiom
table request, the documentation expectations, and the "use established crates"
constraint *before* any code was written. This would have eliminated most
mid-flight course corrections.

### 4. Use `bd` (beads) for task tracking

Beads was initialized but never used for actual issue tracking. For a longer
project, filing tasks as beads issues (with dependencies and labels) would give
persistence across sessions, better traceability, and a record of decisions.

### 5. Integration test design

The current tests skip silently if Ollama isn't running
(`if !ollama_available() { return; }`). In CI this means 0 failures with 0
coverage. Better patterns:
- `#[ignore]` attribute with `cargo test -- --ignored` for integration runs
- Feature gate: `#[cfg(feature = "integration")]`
- CI job that starts Ollama in a container before running tests

### 6. Doc generation in CI

`cargo doc` generates full rustdoc. This should be part of a CI pipeline and
optionally published to GitHub Pages. The doc comments are already in place.

---

## Questions

1. **Target model size** — Should mellea-rust optimize for small models (1-3B,
   needing careful prompt engineering) or assume capable models (8B+, 70B+) where
   simple prompts work? This fundamentally changes the prompt formatting strategy.

2. **mellea-python compatibility** — How important is API parity with the Python
   version? Should Rust developers recognize the same concepts/names, or is a
   fully idiomatic Rust API preferred even if it diverges?

3. **Proc macro priority** — `#[generative]` attribute macro would be the most
   visible ergonomic win. Is this worth investing in for the next iteration, or
   is the builder pattern sufficient?

4. **Provider strategy** — Should mellea-rust add its own provider clients, or
   define traits and provide adapters for existing crates (rig, genai, ollama-rs)?
   The adapter approach is less work but adds dependency coupling.

5. **Upstream interest** — Is there appetite to contribute this upstream to
   `generative-computing/mellea` as an official Rust target, or is this a
   separate project?

6. **Benchmarking** — Should the next iteration include quantitative comparison
   (success rate, retry count, latency) of raw-ollama vs. mellea across different
   models and tasks? This would strengthen the value proposition with data.

---

## What I'd Do Next

### Immediate (next session)

1. **Improve prompt formatting for small models** — The schema-to-example builder
   is naive. Use model-specific strategies: Ollama's `StructuredJson` format for
   models that support it, cleaner example generation for others.

2. **`#[ignore]` integration tests** — Replace the silent skip with proper `#[ignore]`
   so `cargo test` runs unit tests fast, `cargo test -- --ignored` runs integration.

3. **Add `tracing` instrumentation** — Structured logging for every IVR attempt:
   model called, tokens used, validation results, retry count. Zero effort to add,
   massive value for debugging.

4. **CI with GitHub Actions** — Unit tests on every push, integration tests on
   schedule (with Ollama in Docker).

### Short-term (next few iterations)

5. **Typed requirement validators** — The `TypedRequirement<T>` scaffolding exists.
   Wire it into `generate::<T>()` so validators receive `&T` instead of `&str`,
   enabling field-level validation without manual JSON parsing in closures.

6. **`#[generative]` proc macro** — Transform a function signature into a
   `GenerativeSlot`. This is the headline ergonomic feature.

7. **Streaming support** — `generate_stream()` returning `Stream<Item = PartialOutput>`.
   Both ollama-rs and reqwest support SSE streaming.

8. **Multiple sampling strategies** — Best-of-N, majority voting, LLM-as-judge
   (where a second model validates the first). These exist in mellea-python.

### Medium-term

9. **Tool calling** — `MelleaTool` trait with auto-schema from function signature.
   Wire into the Backend trait and IVR loop.

10. **Publish to crates.io** — Once the API stabilizes after 2-3 iterations.
    Claim the `mellea` crate name early.

11. **Benchmarking suite** — Automated comparison of success rate, retry count,
    and latency: raw LLM vs. mellea, across models (granite 3B, llama 8B,
    qwen 32B) and tasks (classification, extraction, generation).
