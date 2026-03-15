//! Agent-facing LCM tools: lcm_grep, lcm_describe, lcm_expand.
//!
//! These tools are injected into the agent loop's `available_tools` list so
//! the LLM can search and expand compressed conversation history at any time.

use openfang_types::tool::ToolDefinition;
use serde_json::{json, Value};

use crate::dag::NodeType;
use crate::store::LcmStore;

/// Return the three LCM tool definitions for injection into the agent loop.
pub fn lcm_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "lcm_grep".to_string(),
            description: concat!(
                "Search the full conversation history — including messages that have been ",
                "compressed out of the active context window — for a keyword or phrase. ",
                "Returns matching excerpts with their node IDs. Use this when you need to ",
                "find something from earlier in the conversation."
            )
            .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for."
                    },
                    "session_id": {
                        "type": "string",
                        "description": "The current session ID."
                    }
                },
                "required": ["query", "session_id"]
            }),
        },
        ToolDefinition {
            name: "lcm_describe".to_string(),
            description: concat!(
                "Show the structure of the conversation DAG for a session. ",
                "Lists all nodes (original messages and summaries) with their IDs, ",
                "types, and content previews. Useful for understanding how much history ",
                "has been compressed and what topics it covers."
            )
            .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID to describe."
                    }
                },
                "required": ["session_id"]
            }),
        },
        ToolDefinition {
            name: "lcm_expand".to_string(),
            description: concat!(
                "Retrieve the full original content of a compressed conversation node. ",
                "Use after lcm_grep or lcm_describe to read the complete text of a ",
                "message or summary that is no longer in the active context window."
            )
            .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The DAG node ID to expand (from lcm_describe or lcm_grep)."
                    }
                },
                "required": ["node_id"]
            }),
        },
    ]
}

/// Returns true if `tool_name` is one of the three LCM tools.
pub fn is_lcm_tool(tool_name: &str) -> bool {
    matches!(tool_name, "lcm_grep" | "lcm_describe" | "lcm_expand")
}

/// Dispatch an LCM tool call. Returns the tool result as a plain string.
pub fn dispatch_lcm_tool(store: &LcmStore, tool_name: &str, input: &Value) -> String {
    match tool_name {
        "lcm_grep" => lcm_grep(store, input),
        "lcm_describe" => lcm_describe(store, input),
        "lcm_expand" => lcm_expand(store, input),
        other => format!("Unknown LCM tool: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

fn lcm_grep(store: &LcmStore, input: &Value) -> String {
    let query = match input["query"].as_str() {
        Some(q) if !q.is_empty() => q,
        _ => return "Error: `query` is required and must be a non-empty string.".to_string(),
    };
    let session_id = match input["session_id"].as_str() {
        Some(s) if !s.is_empty() => s,
        _ => return "Error: `session_id` is required.".to_string(),
    };

    match store.search_nodes(session_id, query) {
        Ok(nodes) if nodes.is_empty() => {
            format!("No matches found for '{}' in session {}.", query, session_id)
        }
        Ok(nodes) => {
            let results: Vec<String> = nodes
                .iter()
                .map(|n| {
                    let type_label = match n.node_type {
                        NodeType::Leaf => {
                            format!("[{}]", n.role.as_deref().unwrap_or("unknown"))
                        }
                        NodeType::Summary => format!("[SUMMARY of {} msgs]", n.children.len()),
                    };
                    let snippet = if n.content.len() > 400 {
                        format!("{}…", &n.content[..400])
                    } else {
                        n.content.clone()
                    };
                    format!("node_id: {}\ntype:    {}\n{}", n.id, type_label, snippet)
                })
                .collect();
            format!(
                "Found {} match(es) for '{}':\n\n{}",
                results.len(),
                query,
                results.join("\n\n---\n\n")
            )
        }
        Err(e) => format!("Error searching history: {e}"),
    }
}

fn lcm_describe(store: &LcmStore, input: &Value) -> String {
    let session_id = match input["session_id"].as_str() {
        Some(s) if !s.is_empty() => s,
        _ => return "Error: `session_id` is required.".to_string(),
    };

    match store.get_session_nodes(session_id) {
        Ok(nodes) if nodes.is_empty() => format!(
            "No LCM nodes found for session `{}`. \
             This session has not been compressed yet.",
            session_id
        ),
        Ok(nodes) => {
            let leaf_count = nodes.iter().filter(|n| n.is_leaf()).count();
            let summary_count = nodes.iter().filter(|n| n.is_summary()).count();
            let mut lines = vec![
                format!("Session: {session_id}"),
                format!(
                    "Nodes: {} total ({leaf_count} original messages, {summary_count} summaries)",
                    nodes.len()
                ),
                String::new(),
            ];
            for node in &nodes {
                let label = match node.node_type {
                    NodeType::Leaf => format!(
                        "LEAF     [{}] idx={}",
                        node.role.as_deref().unwrap_or("?"),
                        node.message_index
                            .map(|i| i.to_string())
                            .unwrap_or_else(|| "?".into())
                    ),
                    NodeType::Summary => {
                        format!("SUMMARY  covers {} messages", node.children.len())
                    }
                };
                let id_short = &node.id[..8.min(node.id.len())];
                lines.push(format!(
                    "  {label} | id={id_short}… | {}",
                    node.preview()
                ));
            }
            lines.join("\n")
        }
        Err(e) => format!("Error describing session: {e}"),
    }
}

fn lcm_expand(store: &LcmStore, input: &Value) -> String {
    let node_id = match input["node_id"].as_str() {
        Some(id) if !id.is_empty() => id,
        _ => return "Error: `node_id` is required.".to_string(),
    };

    match store.get_node(node_id) {
        Ok(Some(node)) => {
            let type_label = match node.node_type {
                NodeType::Leaf => format!(
                    "Original {} message (idx {})",
                    node.role.as_deref().unwrap_or("unknown"),
                    node.message_index
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "?".into())
                ),
                NodeType::Summary => {
                    format!("Summary of {} compressed messages", node.children.len())
                }
            };
            format!(
                "Node ID:  {}\nType:     {type_label}\nTokens:   ~{}\n\n{}",
                node.id, node.token_estimate, node.content
            )
        }
        Ok(None) => format!("Node `{node_id}` not found in LCM store."),
        Err(e) => format!("Error expanding node: {e}"),
    }
}
