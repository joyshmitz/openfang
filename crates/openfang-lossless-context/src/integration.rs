//! High-level integration layer for wiring LCM into the agent loop.
//!
//! [`LosslessContextManager`] is the single entry point that the agent loop
//! (in `openfang-runtime`) uses to:
//!
//! 1. Get the three LCM tool definitions to append to `available_tools`.
//! 2. Dispatch LCM tool calls from the agent.
//! 3. Attempt lossless context-window overflow recovery (instead of
//!    brute-force dropping old messages).

use std::sync::Arc;

use openfang_types::message::Message;
use openfang_types::tool::ToolDefinition;
use tracing::{info, warn};

use crate::compressor::{LcmCompressor, LcmSummarizer};
use crate::store::LcmStore;
use crate::tools::{dispatch_lcm_tool, is_lcm_tool, lcm_tool_definitions};

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
/// One instance is created per agent session and shared (behind an `Arc`)
/// between the compressor and the tool dispatcher.
pub struct LosslessContextManager {
    compressor: LcmCompressor,
    /// Separate read-only handle to the same store (no Arc needed —
    /// SQLite WAL allows concurrent readers).
    dispatch_store: Arc<LcmStore>,
}

impl LosslessContextManager {
    /// Create a new manager backed by the given SQLite database file.
    pub fn new(db_path: &str, summarizer: Box<dyn LcmSummarizer>) -> Result<Self, String> {
        let store =
            LcmStore::open(db_path).map_err(|e| format!("LCM store open error: {e}"))?;
        let dispatch_store = Arc::new(store.clone());

        Ok(Self {
            compressor: LcmCompressor::new(store, summarizer),
            dispatch_store,
        })
    }

    /// Create an in-memory manager (for testing / ephemeral sessions).
    ///
    /// Both the compressor and the dispatcher share the same in-memory
    /// SQLite connection via the cloned `LcmStore` handle.
    pub fn new_in_memory(summarizer: Box<dyn LcmSummarizer>) -> Result<Self, String> {
        let store =
            LcmStore::open_in_memory().map_err(|e| format!("LCM store error: {e}"))?;
        let dispatch_store = Arc::new(store.clone());

        Ok(Self {
            compressor: LcmCompressor::new(store, summarizer),
            dispatch_store,
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
    /// This is a synchronous call — all three tools only do SQLite reads.
    pub fn handle_tool_call(&self, tool_name: &str, input: &serde_json::Value) -> String {
        dispatch_lcm_tool(&self.dispatch_store, tool_name, input)
    }

    // -----------------------------------------------------------------------
    // Overflow recovery
    // -----------------------------------------------------------------------

    /// Attempt lossless overflow recovery by compressing the oldest messages.
    ///
    /// Call this from the context-overflow pipeline (Stage 1 / Stage 2) instead
    /// of the lossy `messages.drain(..remove)` approach.
    ///
    /// `compress_count` — how many messages to compress (typically 10–20).
    pub async fn recover_overflow(
        &self,
        session_id: &str,
        messages: &mut Vec<Message>,
        compress_count: usize,
    ) -> LcmOverflowResult {
        if compress_count == 0 || messages.is_empty() {
            return LcmOverflowResult::NoAction;
        }

        info!(
            session_id = session_id,
            compress_count = compress_count,
            "LCM: starting lossless overflow recovery"
        );

        match self
            .compressor
            .compress_oldest(session_id, messages, compress_count)
            .await
        {
            Ok(0) => LcmOverflowResult::NoAction,
            Ok(n) => {
                info!(
                    session_id = session_id,
                    compressed = n,
                    "LCM: lossless recovery complete"
                );
                LcmOverflowResult::Compressed {
                    messages_compressed: n,
                }
            }
            Err(e) => {
                warn!(
                    session_id = session_id,
                    error = %e,
                    "LCM: compression failed, caller should fall back to lossy trim"
                );
                LcmOverflowResult::Failed(e)
            }
        }
    }
}
