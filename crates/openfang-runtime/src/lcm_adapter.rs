//! Bridge between the agent loop and the lossless context management crate.
//!
//! Provides:
//! - [`LlmBasedSummarizer`] — implements [`LcmSummarizer`] using the agent's
//!   current LLM driver so summaries are generated with the same model.
//! - [`build_context_plugins`] — single entry point used by the agent loop to
//!   construct all context-management plugins for a session.
//!
//! # Design
//!
//! `openfang-lossless-context` defines the `LcmSummarizer` trait but has no
//! dependency on `openfang-runtime` (avoids a circular crate dependency).
//! This module, living in the runtime, provides the concrete implementation
//! that wires the two crates together.
//!
//! The agent loop knows nothing about LCM — it only sees
//! `Vec<Box<dyn ContextPlugin>>` returned by [`build_context_plugins`].

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use openfang_lossless_context::{LcmSummarizer, LosslessContextManager};
use openfang_types::context_plugin::ContextPlugin;
use openfang_types::message::{Message, MessageContent, Role};

use crate::llm_driver::{CompletionRequest, LlmDriver};
use openfang_lossless_context::safe_truncate;

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
                let text = openfang_lossless_context::extract_text(&m.content);
                let snippet = safe_truncate(&text, 800);
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
    session_id: &str,
) -> Option<LosslessContextManager> {
    let summarizer = Box::new(LlmBasedSummarizer::new(driver, model));
    match LosslessContextManager::new_in_memory(summarizer, session_id) {
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
    match LosslessContextManager::new(&db_path_str, summarizer, session_id) {
        Ok(mgr) => Some(mgr),
        Err(e) => {
            tracing::warn!("Failed to initialise disk-backed LCM session at {db_path_str}: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin builder (single entry point for the agent loop)
// ---------------------------------------------------------------------------

/// Build the context-management plugin stack for an agent session.
///
/// The agent loop calls this once at session start and receives an opaque
/// `Vec<Box<dyn ContextPlugin>>`.  It never imports LCM types directly.
///
/// `strategy` comes from `manifest.context.strategy`:
/// - `"lcm"` (default) — DAG-based lossless context management
/// - `"none"` — no context plugins, lossy trim only
pub fn build_context_plugins(
    strategy: &str,
    driver: Arc<dyn LlmDriver>,
    model: &str,
    session_id: &str,
    data_dir: Option<&Path>,
) -> Vec<Box<dyn ContextPlugin>> {
    match strategy {
        "none" => vec![],
        _ => {
            let lcm: Option<Box<dyn ContextPlugin>> = if let Some(dir) = data_dir {
                create_lcm_session_on_disk(driver, model, dir, session_id)
                    .map(|m| Box::new(m) as Box<dyn ContextPlugin>)
            } else {
                create_lcm_session(driver, model, session_id)
                    .map(|m| Box::new(m) as Box<dyn ContextPlugin>)
            };
            lcm.into_iter().collect()
        }
    }
}
