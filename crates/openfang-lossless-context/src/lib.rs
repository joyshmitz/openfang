//! # openfang-lossless-context
//!
//! Lossless conversation context management for the OpenFang Agent OS.
//!
//! Inspired by the `@martian-engineering/lossless-claw` plugin for OpenClaw,
//! this crate replaces the brute-force context-window overflow trimming with a
//! DAG-based compression system:
//!
//! - **Leaf nodes** store original messages verbatim (the agent never loses data).
//! - **Summary nodes** store LLM-generated summaries of compressed message groups.
//! - When the context window fills, the oldest messages are archived as leaves and
//!   replaced by a compact summary — the agent always has something coherent to
//!   read, and can retrieve the full original via `lcm_expand`.
//!
//! ## Three agent tools
//!
//! | Tool | Purpose |
//! |------|---------|
//! | `lcm_grep` | Full-text search across ALL nodes (including compressed history) |
//! | `lcm_describe` | Show DAG structure for a session (what was compressed and when) |
//! | `lcm_expand` | Retrieve the original content of any archived node |
//!
//! ## Integration
//!
//! ```rust,ignore
//! // In openfang-runtime, create a manager once per session:
//! let lcm = LosslessContextManager::new(&db_path, Box::new(MyLlmSummarizer))?;
//!
//! // Append tools so the agent can call them:
//! available_tools.extend(lcm.tool_definitions());
//!
//! // In the context-overflow pipeline (instead of messages.drain):
//! lcm.recover_overflow(&session_id, &mut messages, compress_count).await;
//!
//! // In the tool runner, intercept LCM tool calls:
//! if lcm.owns_tool(&tool_name) {
//!     let result = lcm.handle_tool_call(&tool_name, &input);
//! }
//! ```

pub mod compressor;
pub mod dag;
pub mod integration;
pub mod store;
pub mod tools;

pub use compressor::{LcmCompressor, LcmSummarizer};
pub use dag::{DagNode, NodeType};
pub use integration::{LcmOverflowResult, LosslessContextManager};
pub use store::LcmStore;
pub use tools::{dispatch_lcm_tool, is_lcm_tool, lcm_tool_definitions};
