//! LLM-backed conversation compressor.
//!
//! The [`LcmSummarizer`] trait decouples this crate from `openfang-runtime`.
//! The runtime (or any caller) provides a concrete summarizer that calls
//! whatever LLM driver is configured for the agent.
//!
//! [`LcmCompressor`] takes the oldest N messages, archives each as a leaf
//! DAG node, calls the summarizer, and stores the resulting summary node.
//! It then injects a synthetic "context summary" message in place of the
//! compressed messages so the agent loop sees a compact but informative
//! replacement.

use async_trait::async_trait;
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use tracing::{debug, warn};

use crate::dag::DagNode;
use crate::store::LcmStore;

/// Abstraction over an LLM call for generating summaries.
///
/// Implement this trait in `openfang-runtime` using the agent's current
/// LLM driver so the compressor can generate coherent summaries without
/// a circular crate dependency.
#[async_trait]
pub trait LcmSummarizer: Send + Sync {
    /// Produce a concise summary of the provided messages.
    ///
    /// The implementation should call the LLM with a short system prompt
    /// asking it to summarise the conversation, then return the raw summary
    /// text. If the call fails, return an `Err` and the compressor will
    /// fall back to a concatenation-based summary.
    async fn summarize(&self, messages: &[Message]) -> Result<String, String>;
}

/// Compresses the oldest messages in a session into DAG summary nodes.
pub struct LcmCompressor {
    pub(crate) store: LcmStore,
    summarizer: Box<dyn LcmSummarizer>,
}

impl LcmCompressor {
    /// Create a new compressor backed by the given store and summarizer.
    pub fn new(store: LcmStore, summarizer: Box<dyn LcmSummarizer>) -> Self {
        Self { store, summarizer }
    }

    /// Compress the oldest `count` messages in `messages` into a single summary node.
    ///
    /// The compressed messages are:
    /// 1. Stored as individual leaf DAG nodes (original content preserved).
    /// 2. Summarised via the LLM summarizer (or fallback concatenation).
    /// 3. Replaced in-place with a single synthetic "CONTEXT SUMMARY" user message.
    ///
    /// Returns the number of messages actually compressed.
    pub async fn compress_oldest(
        &self,
        session_id: &str,
        messages: &mut Vec<Message>,
        count: usize,
    ) -> Result<usize, String> {
        if count == 0 || messages.is_empty() {
            return Ok(0);
        }

        let actual = count.min(messages.len());
        let to_compress: Vec<Message> = messages.drain(..actual).collect();

        debug!(
            session_id = session_id,
            compressing = actual,
            "LCM: archiving {} messages as leaf nodes",
            actual
        );

        // Archive each message as a leaf node.
        let mut leaf_ids = Vec::with_capacity(actual);
        for (idx, msg) in to_compress.iter().enumerate() {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let content = extract_text(&msg.content);
            let leaf = DagNode::new_leaf(session_id, role, &content, idx as u64);
            let leaf_id = leaf.id.clone();
            self.store
                .insert_node(&leaf)
                .map_err(|e| format!("lcm store error: {e}"))?;
            leaf_ids.push(leaf_id);
        }

        // Summarise via the LLM (fallback to concatenation on error).
        let summary_text = match self.summarizer.summarize(&to_compress).await {
            Ok(s) => {
                debug!("LCM: LLM summary produced ({} chars)", s.len());
                s
            }
            Err(e) => {
                warn!("LCM: summarizer failed ({e}), using concatenation fallback");
                fallback_summary(&to_compress)
            }
        };

        // Store the summary node.
        let summary = DagNode::new_summary(session_id, &summary_text, leaf_ids);
        self.store
            .insert_node(&summary)
            .map_err(|e| format!("lcm store error: {e}"))?;

        // Update session root metadata.
        let remaining = messages.len() as u64;
        self.store
            .upsert_session_root(session_id, remaining)
            .map_err(|e| format!("lcm store error: {e}"))?;

        // Inject a synthetic context-summary message at the front.
        let summary_msg = Message {
            role: Role::User,
            content: MessageContent::Text(format!(
                "[CONTEXT SUMMARY — {actual} earlier messages compressed]\n\n{summary_text}"
            )),
        };
        messages.insert(0, summary_msg);

        Ok(actual)
    }
}

/// Extract displayable text from a message's content.
pub(crate) fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                ContentBlock::ToolResult { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Concatenation-based fallback summary (used when the LLM summarizer fails).
fn fallback_summary(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
            };
            let text = extract_text(&m.content);
            let snippet = &text[..200.min(text.len())];
            format!("[{role}]: {snippet}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::LcmStore;
    use openfang_types::message::MessageContent;

    struct EchoSummarizer;

    #[async_trait]
    impl LcmSummarizer for EchoSummarizer {
        async fn summarize(&self, messages: &[Message]) -> Result<String, String> {
            Ok(format!("Summary of {} messages.", messages.len()))
        }
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: MessageContent::Text(text.to_string()),
        }
    }

    #[tokio::test]
    async fn compress_oldest_replaces_messages() {
        let store = LcmStore::open_in_memory().unwrap();
        let compressor = LcmCompressor::new(store, Box::new(EchoSummarizer));

        let mut messages = vec![
            user_msg("msg0"),
            user_msg("msg1"),
            user_msg("msg2"),
            user_msg("msg3"),
        ];

        let compressed = compressor
            .compress_oldest("sess-test", &mut messages, 2)
            .await
            .unwrap();

        assert_eq!(compressed, 2);
        // 2 original remain + 1 injected summary = 3
        assert_eq!(messages.len(), 3);

        let first = extract_text(&messages[0].content);
        assert!(first.contains("CONTEXT SUMMARY"));
        assert!(first.contains("Summary of 2 messages."));
    }
}
