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
                    }
                },
                "required": ["query"]
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
                "properties": {},
                "required": []
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

/// Dispatch an LCM tool call. Returns `Ok(content)` on success or `Err(message)` on error.
///
/// `Err` strings are prefixed with `"Error: "` so that providers which ignore
/// the `is_error` flag (OpenAI, Gemini) still surface the failure to the model.
pub fn dispatch_lcm_tool(store: &LcmStore, tool_name: &str, input: &Value) -> Result<String, String> {
    match tool_name {
        "lcm_grep" => lcm_grep(store, input),
        "lcm_describe" => lcm_describe(store, input),
        "lcm_expand" => lcm_expand(store, input),
        other => Err(format!("Error: unknown LCM tool: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

fn lcm_grep(store: &LcmStore, input: &Value) -> Result<String, String> {
    let query = match input["query"].as_str() {
        Some(q) if !q.is_empty() => q,
        _ => return Err("Error: `query` is required and must be a non-empty string.".to_string()),
    };
    let session_id = match input["session_id"].as_str() {
        Some(s) if !s.is_empty() => s,
        _ => return Err("Error: `session_id` is required.".to_string()),
    };

    match store.search_nodes(session_id, query) {
        Ok(nodes) if nodes.is_empty() => {
            Ok(format!("No matches found for '{}' in session {}.", query, session_id))
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
                        format!("{}…", crate::safe_truncate(&n.content, 400))
                    } else {
                        n.content.clone()
                    };
                    format!("node_id: {}\ntype:    {}\n{}", n.id, type_label, snippet)
                })
                .collect();
            Ok(format!(
                "Found {} match(es) for '{}':\n\n{}",
                results.len(),
                query,
                results.join("\n\n---\n\n")
            ))
        }
        Err(e) => Err(format!("Error: searching history: {e}")),
    }
}

fn lcm_describe(store: &LcmStore, input: &Value) -> Result<String, String> {
    let session_id = match input["session_id"].as_str() {
        Some(s) if !s.is_empty() => s,
        _ => return Err("Error: `session_id` is required.".to_string()),
    };

    match store.get_session_nodes(session_id) {
        Ok(nodes) if nodes.is_empty() => Ok(format!(
            "No LCM nodes found for session `{}`. \
             This session has not been compressed yet.",
            session_id
        )),
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
                        "LEAF     [{}] idx={} d={}",
                        node.role.as_deref().unwrap_or("?"),
                        node.message_index
                            .map(|i| i.to_string())
                            .unwrap_or_else(|| "?".into()),
                        node.depth
                    ),
                    NodeType::Summary => {
                        format!("SUMMARY  d={} covers {} messages", node.depth, node.children.len())
                    }
                };
                let id_short = &node.id[..8.min(node.id.len())];
                lines.push(format!(
                    "  {label} | id={id_short}… | {}",
                    node.preview()
                ));
            }
            Ok(lines.join("\n"))
        }
        Err(e) => Err(format!("Error: describing session: {e}")),
    }
}

fn lcm_expand(store: &LcmStore, input: &Value) -> Result<String, String> {
    let node_id = match input["node_id"].as_str() {
        Some(id) if !id.is_empty() => id,
        _ => return Err("Error: `node_id` is required.".to_string()),
    };

    match store.get_node(node_id) {
        Ok(Some(node)) => {
            let type_label = match node.node_type {
                NodeType::Leaf => format!(
                    "Original {} message (idx {}, depth {})",
                    node.role.as_deref().unwrap_or("unknown"),
                    node.message_index
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "?".into()),
                    node.depth
                ),
                NodeType::Summary => {
                    format!("Summary (depth {}) of {} compressed messages", node.depth, node.children.len())
                }
            };
            Ok(format!(
                "Node ID:  {}\nType:     {type_label}\nTokens:   ~{}\n\n{}",
                node.id, node.token_estimate, node.content
            ))
        }
        Ok(None) => Err(format!("Error: node `{node_id}` not found in LCM store.")),
        Err(e) => Err(format!("Error: expanding node: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DagNode;
    use crate::store::LcmStore;
    use serde_json::json;

    #[test]
    fn expand_nonexistent_node() {
        let store = LcmStore::open_in_memory().unwrap();
        let result = dispatch_lcm_tool(&store, "lcm_expand", &json!({"node_id": "no-such-id"}));
        let err = result.unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn describe_with_multiple_summaries() {
        let store = LcmStore::open_in_memory().unwrap();

        let leaf1 = DagNode::new_leaf("sess-d", "user", "hello", 0);
        let leaf2 = DagNode::new_leaf("sess-d", "assistant", "hi there", 1);
        let sum1 = DagNode::new_summary("sess-d", "first batch", vec![leaf1.id.clone(), leaf2.id.clone()], 1);

        let leaf3 = DagNode::new_leaf("sess-d", "user", "another msg", 2);
        let sum2 = DagNode::new_summary("sess-d", "second batch", vec![leaf3.id.clone()], 1);

        for node in [&leaf1, &leaf2, &sum1, &leaf3, &sum2] {
            store.insert_node(node).unwrap();
        }

        let result = dispatch_lcm_tool(&store, "lcm_describe", &json!({"session_id": "sess-d"})).unwrap();
        assert!(result.contains("3 original messages"));
        assert!(result.contains("2 summaries"));
        assert!(result.contains("LEAF"));
        assert!(result.contains("SUMMARY"));
    }

    #[test]
    fn grep_with_unicode_snippet_truncation() {
        let store = LcmStore::open_in_memory().unwrap();
        // Content > 400 bytes — will trigger the truncation path
        let long_text = "Я".repeat(300); // 300 × 2 bytes = 600 bytes
        let node = DagNode::new_leaf("sess-g", "user", &long_text, 0);
        store.insert_node(&node).unwrap();

        // Should not panic on multi-byte truncation
        let result = dispatch_lcm_tool(
            &store,
            "lcm_grep",
            &json!({"query": "Я", "session_id": "sess-g"}),
        ).unwrap();
        assert!(result.contains("1 match"));
    }

    #[test]
    fn grep_missing_query() {
        let store = LcmStore::open_in_memory().unwrap();
        let result = dispatch_lcm_tool(
            &store,
            "lcm_grep",
            &json!({"session_id": "sess-x"}),
        );
        assert!(result.is_err());
    }
}
