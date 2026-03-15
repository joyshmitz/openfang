//! Context management plugin trait for the agent loop.
//!
//! Defines a pluggable interface for context-window strategies so that the
//! agent loop does not hardcode any single approach.  Implementations can
//! range from simple sliding-window trimming to DAG-based lossless archival
//! (LCM) to RAG-backed vector-store retrieval.
//!
//! The agent loop interacts with context plugins at three points:
//!
//! 1. **Tool registration** — [`ContextPlugin::tool_definitions`] exposes
//!    plugin-specific tools to the LLM.
//! 2. **Tool dispatch** — [`ContextPlugin::handle_tool_call`] intercepts
//!    tool calls that belong to the plugin.
//! 3. **Overflow recovery** — [`ContextPlugin::on_context_overflow`] is
//!    called before the lossy trim pipeline when context usage exceeds the
//!    configured threshold.

use async_trait::async_trait;

use crate::message::Message;
use crate::tool::ToolDefinition;

/// A pluggable context-management strategy injected into the agent loop.
///
/// One or more plugins can be active per agent session.  The agent loop
/// iterates over all registered plugins for each of the three touch-points
/// (tools, dispatch, overflow).
#[async_trait]
pub trait ContextPlugin: Send + Sync {
    /// Human-readable name for logging and diagnostics (e.g. `"lcm"`, `"sliding_window"`).
    fn name(&self) -> &str;

    /// Tool definitions to expose to the LLM.
    ///
    /// Called once at session setup; the returned definitions are appended to
    /// `available_tools`.
    fn tool_definitions(&self) -> Vec<ToolDefinition>;

    /// Try to handle a tool call.
    ///
    /// Returns `Some((content, is_error))` if this plugin owns `tool_name`,
    /// or `None` to let the next plugin (or the default tool runner) handle it.
    fn handle_tool_call(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
    ) -> Option<(String, bool)>;

    /// Called before each LLM call when context usage exceeds `threshold_pct`
    /// of the context window.
    ///
    /// The plugin may mutate `messages` in place (e.g. compress, archive,
    /// drop) to bring usage back below the threshold.  Returns the number of
    /// messages affected (0 = no action).
    async fn on_context_overflow(
        &self,
        messages: &mut Vec<Message>,
        threshold_pct: f64,
    ) -> Result<usize, String>;
}
