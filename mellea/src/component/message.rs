use crate::core::ChatMessage;

/// Extension methods for ChatMessage.
pub trait ChatMessageExt {
    fn is_user(&self) -> bool;
    fn is_assistant(&self) -> bool;
    fn is_system(&self) -> bool;
}

impl ChatMessageExt for ChatMessage {
    fn is_user(&self) -> bool {
        self.role == crate::core::Role::User
    }

    fn is_assistant(&self) -> bool {
        self.role == crate::core::Role::Assistant
    }

    fn is_system(&self) -> bool {
        self.role == crate::core::Role::System
    }
}
