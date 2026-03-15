//! High-level integration layer for wiring LCM into the agent loop.
//!
//! [`LosslessContextManager`] is the single entry point that the agent loop
//! (in `openfang-runtime`) uses to:
//!
//! 1. Get the three LCM tool definitions to append to `available_tools`.
//! 2. Dispatch LCM tool calls from the agent.
//! 3. Attempt lossless context-window overflow recovery (instead of
//!    brute-force dropping old messages).
//!
//! It also implements [`ContextPlugin`] so it can be used as a generic,
//! swappable context strategy in the agent loop's plugin pipeline.

use std::sync::Arc;

use async_trait::async_trait;
use openfang_types::context_plugin::ContextPlugin;
use openfang_types::message::Message;
use openfang_types::tool::ToolDefinition;
use tracing::{info, warn};

use crate::compressor::{LcmCompressor, LcmSummarizer};
use crate::store::LcmStore;
use crate::tools::{dispatch_lcm_tool, is_lcm_tool, lcm_tool_definitions};

/// Default number of oldest messages to compress per overflow recovery pass.
const DEFAULT_COMPRESS_COUNT: usize = 10;

/// Result of an LCM overflow-recovery attempt.
#[derive(Debug)]
pub enum LcmOverflowResult {
    /// Lossless recovery succeeded — N messages were compressed.
    Compressed { messages_compressed: usize },
    /// Nothing needed to be done (context is still within limits).
    NoAction,
    /// Recovery failed; caller should fall back to lossy trimming.
    Failed(String),
}

/// The main integration struct.
///
/// One instance is created per agent session. Stores the `session_id` so that
/// LCM tool calls (lcm_grep, lcm_describe) auto-receive it — the model does
/// not need to know or guess the current session ID.
pub struct LosslessContextManager {
    compressor: LcmCompressor,
    dispatch_store: Arc<LcmStore>,
    session_id: String,
}

impl LosslessContextManager {
    /// Create a new manager backed by the given SQLite database file.
    pub fn new(
        db_path: &str,
        summarizer: Box<dyn LcmSummarizer>,
        session_id: &str,
    ) -> Result<Self, String> {
        let store =
            LcmStore::open(db_path).map_err(|e| format!("LCM store open error: {e}"))?;
        let dispatch_store = Arc::new(store.clone());

        Ok(Self {
            compressor: LcmCompressor::new(store, summarizer),
            dispatch_store,
            session_id: session_id.to_string(),
        })
    }

    /// Create an in-memory manager (for testing / ephemeral sessions).
    pub fn new_in_memory(
        summarizer: Box<dyn LcmSummarizer>,
        session_id: &str,
    ) -> Result<Self, String> {
        let store =
            LcmStore::open_in_memory().map_err(|e| format!("LCM store error: {e}"))?;
        let dispatch_store = Arc::new(store.clone());

        Ok(Self {
            compressor: LcmCompressor::new(store, summarizer),
            dispatch_store,
            session_id: session_id.to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Tool registration
    // -----------------------------------------------------------------------

    /// Tool definitions to append to the agent's `available_tools` list.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        lcm_tool_definitions()
    }

    /// Returns true if `tool_name` should be handled by LCM (not by the
    /// normal tool runner).
    pub fn owns_tool(&self, tool_name: &str) -> bool {
        is_lcm_tool(tool_name)
    }

    // -----------------------------------------------------------------------
    // Tool dispatch
    // -----------------------------------------------------------------------

    /// Handle an LCM tool call from the agent loop.
    ///
    /// Auto-injects the current `session_id` into the input so the model does
    /// not need to know it. Returns `(content, is_error)`.
    pub fn handle_tool_call(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> (String, bool) {
        let enriched = self.inject_session_id(input);
        match dispatch_lcm_tool(&self.dispatch_store, tool_name, &enriched) {
            Ok(content) => (content, false),
            Err(content) => (content, true),
        }
    }

    /// Coerce `input` to an object (if null/non-object) and inject `session_id`.
    ///
    /// Some providers (OpenAI, Gemini) emit `null` instead of `{}` for
    /// zero-arg tools like `lcm_describe`, so we must handle that case.
    fn inject_session_id(&self, input: &serde_json::Value) -> serde_json::Value {
        let mut obj = match input {
            serde_json::Value::Object(m) => serde_json::Value::Object(m.clone()),
            _ => serde_json::json!({}),
        };
        obj.as_object_mut().unwrap().insert(
            "session_id".to_string(),
            serde_json::Value::String(self.session_id.clone()),
        );
        obj
    }

    // -----------------------------------------------------------------------
    // Overflow recovery
    // -----------------------------------------------------------------------

    /// The session ID this manager was created for.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Attempt lossless overflow recovery by compressing the oldest messages.
    ///
    /// Call this from the context-overflow pipeline (Stage 1 / Stage 2) instead
    /// of the lossy `messages.drain(..remove)` approach.
    ///
    /// `compress_count` — how many messages to compress (typically 10–20).
    pub async fn recover_overflow(
        &self,
        messages: &mut Vec<Message>,
        compress_count: usize,
    ) -> LcmOverflowResult {
        if compress_count == 0 || messages.is_empty() {
            return LcmOverflowResult::NoAction;
        }

        info!(
            session_id = %self.session_id,
            compress_count = compress_count,
            "LCM: starting lossless overflow recovery"
        );

        match self
            .compressor
            .compress_oldest(&self.session_id, messages, compress_count)
            .await
        {
            Ok(result) if result.real_count == 0 => LcmOverflowResult::NoAction,
            Ok(result) => {
                info!(
                    session_id = %self.session_id,
                    compressed = result.real_count,
                    "LCM: lossless recovery complete"
                );
                LcmOverflowResult::Compressed {
                    messages_compressed: result.real_count,
                }
            }
            Err(e) => {
                warn!(
                    session_id = %self.session_id,
                    error = %e,
                    "LCM: compression failed, caller should fall back to lossy trim"
                );
                LcmOverflowResult::Failed(e)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContextPlugin implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ContextPlugin for LosslessContextManager {
    fn name(&self) -> &str {
        "lcm"
    }

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        lcm_tool_definitions()
    }

    fn handle_tool_call(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> Option<(String, bool)> {
        if !is_lcm_tool(tool_name) {
            return None;
        }
        let enriched = self.inject_session_id(input);
        Some(match dispatch_lcm_tool(&self.dispatch_store, tool_name, &enriched) {
            Ok(content) => (content, false),
            Err(content) => (content, true),
        })
    }

    async fn on_context_overflow(
        &self,
        messages: &mut Vec<Message>,
        _threshold_pct: f64,
    ) -> Result<usize, String> {
        match self.recover_overflow(messages, DEFAULT_COMPRESS_COUNT).await {
            LcmOverflowResult::Compressed { messages_compressed } => Ok(messages_compressed),
            LcmOverflowResult::NoAction => Ok(0),
            LcmOverflowResult::Failed(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::LcmSummarizer;
    use openfang_types::message::Message;
    use serde_json::json;

    struct StubSummarizer;

    #[async_trait]
    impl LcmSummarizer for StubSummarizer {
        async fn summarize(&self, _messages: &[Message]) -> Result<String, String> {
            Ok("stub".to_string())
        }
    }

    #[test]
    fn handle_tool_call_returns_is_error_for_error_prefix() {
        let mgr =
            LosslessContextManager::new_in_memory(Box::new(StubSummarizer), "sess-err").unwrap();

        // lcm_expand with nonexistent node → "Error: node ... not found" → is_error = true
        let (content, is_error) = mgr.handle_tool_call("lcm_expand", &json!({"node_id": "nope"}));
        assert!(is_error, "Expected is_error=true for not-found. Got: {content}");
        assert!(content.contains("not found"));

        // lcm_grep with missing required param → "Error: ..." → is_error = true
        let (content, is_error) = mgr.handle_tool_call("lcm_grep", &json!({}));
        assert!(is_error, "Expected is_error=true for missing param. Got: {content}");

        // lcm_describe with valid (empty) session → informational → is_error = false
        let (content, is_error) = mgr.handle_tool_call("lcm_describe", &json!({}));
        assert!(!is_error, "Expected is_error=false for empty describe. Got: {content}");
    }

    #[test]
    fn handle_tool_call_auto_injects_session_id() {
        let mgr =
            LosslessContextManager::new_in_memory(Box::new(StubSummarizer), "sess-inject").unwrap();

        // lcm_describe with no session_id in input — should auto-inject and not error
        let (content, is_error) = mgr.handle_tool_call("lcm_describe", &json!({}));
        assert!(!is_error);
        // Should mention the session (either "No LCM nodes" or the session id)
        assert!(content.contains("sess-inject"));
    }
}
