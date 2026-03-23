use super::message::ChatMessage;
use super::output::ModelOutput;

/// A single conversation turn: an input message and its optional output.
#[derive(Debug, Clone)]
pub struct Turn {
    pub input: ChatMessage,
    pub output: Option<ModelOutput>,
}

/// Immutable conversation history.
/// Each operation returns a new Context — the original is unchanged.
#[derive(Debug, Clone, Default)]
pub struct Context {
    turns: Vec<Turn>,
}

impl Context {
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }

    /// Append a turn, returning a new Context.
    pub fn push(&self, turn: Turn) -> Self {
        let mut turns = self.turns.clone();
        turns.push(turn);
        Self { turns }
    }

    /// Get the most recent model output, if any.
    pub fn last_output(&self) -> Option<&ModelOutput> {
        self.turns.iter().rev().find_map(|t| t.output.as_ref())
    }

    /// Flatten the context into a list of ChatMessages suitable for an API call.
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut msgs = Vec::new();
        for turn in &self.turns {
            msgs.push(turn.input.clone());
            if let Some(ref output) = turn.output {
                msgs.push(ChatMessage::assistant(&output.content));
            }
        }
        msgs
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::Role;
    use crate::core::output::Usage;

    #[test]
    fn test_context_immutability() {
        let ctx = Context::new();
        let ctx2 = ctx.push(Turn {
            input: ChatMessage::user("hello"),
            output: None,
        });
        assert_eq!(ctx.len(), 0);
        assert_eq!(ctx2.len(), 1);
    }

    #[test]
    fn test_messages_roundtrip() {
        let ctx = Context::new()
            .push(Turn {
                input: ChatMessage::user("hi"),
                output: Some(ModelOutput {
                    content: "hello!".to_string(),
                    model: "test".to_string(),
                    usage: Usage::default(),
                }),
            })
            .push(Turn {
                input: ChatMessage::user("how are you?"),
                output: None,
            });

        let msgs = ctx.messages();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::User);
    }

    #[test]
    fn test_last_output() {
        let ctx = Context::new().push(Turn {
            input: ChatMessage::user("test"),
            output: Some(ModelOutput {
                content: "response".to_string(),
                model: "m".to_string(),
                usage: Usage::default(),
            }),
        });
        assert_eq!(ctx.last_output().unwrap().content, "response");
    }
}
