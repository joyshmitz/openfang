//! Bridge between the agent loop and the lossless context management crate.
//!
//! Provides:
//! - [`LlmBasedSummarizer`] — implements [`LcmSummarizer`] using the agent's
//!   current LLM driver so summaries are generated with the same model.
//! - [`create_lcm_session`] — factory used by `run_agent_loop` to build a
//!   per-session [`LosslessContextManager`].
//!
//! # Design
//!
//! `openfang-lossless-context` defines the `LcmSummarizer` trait but has no
//! dependency on `openfang-runtime` (avoids a circular crate dependency).
//! This module, living in the runtime, provides the concrete implementation
//! that wires the two crates together.

use std::sync::Arc;

use async_trait::async_trait;
use openfang_lossless_context::{LcmSummarizer, LosslessContextManager};
use openfang_types::message::{Message, MessageContent, Role};

use crate::llm_driver::{CompletionRequest, LlmDriver};
use openfang_types::message::ContentBlock;

// ---------------------------------------------------------------------------
// LlmBasedSummarizer
// ---------------------------------------------------------------------------

/// An [`LcmSummarizer`] that calls the agent's current LLM driver.
pub struct LlmBasedSummarizer {
    driver: Arc<dyn LlmDriver>,
    model: String,
}

impl LlmBasedSummarizer {
    pub fn new(driver: Arc<dyn LlmDriver>, model: impl Into<String>) -> Self {
        Self {
            driver,
            model: model.into(),
        }
    }
}

#[async_trait]
impl LcmSummarizer for LlmBasedSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<String, String> {
        let system = "You are a conversation summarizer. \
            Given the following conversation excerpt, produce a concise, \
            information-dense summary that preserves all key facts, \
            decisions, names, numbers, and technical details. \
            Write in third-person past tense. Maximum 350 words."
            .to_string();

        // Build a text representation of the messages to summarize.
        let user_text: String = messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::System => "System",
                };
                let text = msg_to_text(m);
                let snippet = &text[..800.min(text.len())];
                format!("[{i}] {role}: {snippet}")
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let request = CompletionRequest {
            model: self.model.clone(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text(user_text),
            }],
            tools: vec![],
            max_tokens: 512,
            temperature: 0.2,
            system: Some(system),
            thinking: None,
        };

        let response = self
            .driver
            .complete(request)
            .await
            .map_err(|e| format!("LCM summarizer LLM call failed: {e}"))?;

        Ok(response.text())
    }
}

// ---------------------------------------------------------------------------
// Text extraction helper
// ---------------------------------------------------------------------------

fn msg_to_text(msg: &Message) -> String {
    match &msg.content {
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

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create an in-memory [`LosslessContextManager`] for a single agent session.
///
/// Used as fallback when no kernel data directory is available (e.g. tests or
/// embedded contexts). Data does not persist across daemon restarts.
///
/// Returns `None` (and logs a warning) if the manager cannot be initialised
/// rather than panicking, so that LCM failure never crashes the agent loop.
pub fn create_lcm_session(
    driver: Arc<dyn LlmDriver>,
    model: &str,
) -> Option<LosslessContextManager> {
    let summarizer = Box::new(LlmBasedSummarizer::new(driver, model));
    match LosslessContextManager::new_in_memory(summarizer) {
        Ok(mgr) => Some(mgr),
        Err(e) => {
            tracing::warn!("Failed to initialise LCM session: {e}");
            None
        }
    }
}

/// Create a disk-backed [`LosslessContextManager`] scoped to a single session.
///
/// The SQLite database is stored at `<data_dir>/lcm/<session_id>.db` so that
/// the compressed message DAG survives daemon restarts.
///
/// Falls back to `None` if the directory cannot be created or the store cannot
/// be opened — LCM failure must never crash the agent loop.
pub fn create_lcm_session_on_disk(
    driver: Arc<dyn LlmDriver>,
    model: &str,
    data_dir: &std::path::Path,
    session_id: &str,
) -> Option<LosslessContextManager> {
    let lcm_dir = data_dir.join("lcm");
    if let Err(e) = std::fs::create_dir_all(&lcm_dir) {
        tracing::warn!("Failed to create LCM directory {lcm_dir:?}: {e}");
        return None;
    }
    let db_path = lcm_dir.join(format!("{session_id}.db"));
    let db_path_str = match db_path.to_str() {
        Some(s) => s.to_owned(),
        None => {
            tracing::warn!("LCM db path is not valid UTF-8: {db_path:?}");
            return None;
        }
    };
    let summarizer = Box::new(LlmBasedSummarizer::new(driver, model));
    match LosslessContextManager::new(&db_path_str, summarizer) {
        Ok(mgr) => Some(mgr),
        Err(e) => {
            tracing::warn!("Failed to initialise disk-backed LCM session at {db_path_str}: {e}");
            None
        }
    }
}
