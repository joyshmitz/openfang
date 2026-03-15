//! DAG node types for lossless conversation context management.
//!
//! The conversation history is represented as a DAG where:
//! - Leaf nodes = original unmodified messages
//! - Summary nodes = LLM-generated summaries of one or more child nodes
//!
//! When the context window fills up, the oldest messages are compressed into
//! a Summary node. The original text is preserved in the leaf nodes, which
//! remain searchable via `lcm_grep` and retrievable via `lcm_expand`.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The type of a DAG node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// Original unmodified message — leaf in the DAG.
    Leaf,
    /// LLM-generated summary of child messages — internal node.
    Summary,
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Leaf => write!(f, "leaf"),
            NodeType::Summary => write!(f, "summary"),
        }
    }
}

/// A node in the lossless conversation DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    /// Unique node identifier (UUID v4).
    pub id: String,
    /// The session this node belongs to.
    pub session_id: String,
    /// Parent summary node, if any.
    pub parent_id: Option<String>,
    /// Whether this is a leaf (original) or summary (compressed) node.
    pub node_type: NodeType,
    /// For Leaf: the original message role ("user" / "assistant" / "tool").
    /// For Summary: None.
    pub role: Option<String>,
    /// The text content.
    /// For Leaf: original message text.
    /// For Summary: the LLM-generated summary text.
    pub content: String,
    /// Original 0-based position index in the session (Leaf only).
    pub message_index: Option<u64>,
    /// IDs of child nodes (Summary only — the messages this node summarises).
    pub children: Vec<String>,
    /// Rough token count estimate (content.len() / 4).
    pub token_estimate: u64,
    /// Unix timestamp of node creation.
    pub created_at: i64,
}

impl DagNode {
    /// Create a leaf node from an original message.
    pub fn new_leaf(
        session_id: &str,
        role: &str,
        content: &str,
        message_index: u64,
    ) -> Self {
        let token_estimate = (content.len() as u64 / 4).max(1);
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            parent_id: None,
            node_type: NodeType::Leaf,
            role: Some(role.to_string()),
            content: content.to_string(),
            message_index: Some(message_index),
            children: Vec::new(),
            token_estimate,
            created_at: Utc::now().timestamp(),
        }
    }

    /// Create a summary node from child node IDs and an LLM-generated summary.
    pub fn new_summary(
        session_id: &str,
        summary: &str,
        children: Vec<String>,
    ) -> Self {
        let token_estimate = (summary.len() as u64 / 4).max(1);
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            parent_id: None,
            node_type: NodeType::Summary,
            role: None,
            content: summary.to_string(),
            message_index: None,
            children,
            token_estimate,
            created_at: Utc::now().timestamp(),
        }
    }

    /// Returns true if this is a leaf (original) node.
    pub fn is_leaf(&self) -> bool {
        self.node_type == NodeType::Leaf
    }

    /// Returns true if this is a summary (compressed) node.
    pub fn is_summary(&self) -> bool {
        self.node_type == NodeType::Summary
    }

    /// A short preview of the content (first 80 chars).
    pub fn preview(&self) -> &str {
        let end = self.content.len().min(80);
        &self.content[..end]
    }
}
