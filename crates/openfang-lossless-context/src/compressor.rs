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

/// Outcome of a single compression pass.
#[derive(Debug, Default)]
pub struct CompressResult {
    /// Number of real (non-synthetic) messages archived.
    pub real_count: usize,
    /// ID of the summary node that was created, if any.
    pub summary_node_id: Option<String>,
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

    // -----------------------------------------------------------------------
    // Decomposed building blocks (reusable by V2 multi-depth condensation)
    // -----------------------------------------------------------------------

    /// Archive messages as leaf DAG nodes. Returns the leaf node IDs.
    ///
    /// Uses `store.leaf_count()` as the base offset so `message_index` values
    /// are globally unique across multiple compression passes.
    pub fn archive_leaves(
        &self,
        session_id: &str,
        messages: &[Message],
    ) -> Result<Vec<String>, String> {
        let base_idx = self
            .store
            .leaf_count(session_id)
            .map_err(|e| format!("lcm store error: {e}"))?;

        let mut leaf_ids = Vec::with_capacity(messages.len());
        for (idx, msg) in messages.iter().enumerate() {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let content = extract_text(&msg.content);
            let leaf = DagNode::new_leaf(session_id, role, &content, base_idx + idx as u64);
            let leaf_id = leaf.id.clone();
            self.store
                .insert_node(&leaf)
                .map_err(|e| format!("lcm store error: {e}"))?;
            leaf_ids.push(leaf_id);
        }
        Ok(leaf_ids)
    }

    /// Call the summarizer and store the resulting summary node.
    ///
    /// `depth` is 1 for summaries of leaves, 2+ for summaries of summaries.
    /// Returns the stored `DagNode`.
    pub async fn create_summary(
        &self,
        session_id: &str,
        summarizer_input: &[Message],
        child_ids: Vec<String>,
        depth: u32,
    ) -> Result<DagNode, String> {
        let summary_text = match self.summarizer.summarize(summarizer_input).await {
            Ok(s) => {
                debug!("LCM: LLM summary produced ({} chars)", s.len());
                s
            }
            Err(e) => {
                warn!("LCM: summarizer failed ({e}), using concatenation fallback");
                fallback_summary(summarizer_input)
            }
        };

        let summary = DagNode::new_summary(session_id, &summary_text, child_ids, depth);
        self.store
            .insert_node(&summary)
            .map_err(|e| format!("lcm store error: {e}"))?;
        Ok(summary)
    }

    // -----------------------------------------------------------------------
    // Top-level orchestrator
    // -----------------------------------------------------------------------

    /// Compress the oldest `count` messages into a single summary node.
    ///
    /// Synthetic summary messages from prior compressions within the same
    /// turn are identified by a `lcm_node=<uuid>` marker embedded in their
    /// content.  The marker must reference a **summary** node in this
    /// session's store; leaf IDs and cross-session IDs are rejected.
    ///
    /// # Cross-turn limitation
    ///
    /// The marker only survives if the **host** persists the modified
    /// `messages` vec between turns.  If the host rebuilds messages from
    /// session storage (dropping the synthetic summary), the next turn's
    /// overflow will archive the same raw turns as new leaves.  This is
    /// safe (no corruption, no duplicate node IDs) but produces unlinked
    /// DAG segments.  To get full cross-turn linking, the host must persist
    /// the synthetic `[LCM lcm_node=…]` message as part of the conversation
    /// history.
    ///
    /// Returns a [`CompressResult`] with the count and new summary node ID.
    pub async fn compress_oldest(
        &self,
        session_id: &str,
        messages: &mut Vec<Message>,
        count: usize,
    ) -> Result<CompressResult, String> {
        if count == 0 || messages.is_empty() {
            return Ok(CompressResult::default());
        }

        let actual = count.min(messages.len());
        let drained: Vec<Message> = messages.drain(..actual).collect();

        // Partition: detect prior synthetic summaries by embedded marker.
        // Only markers that reference a node in THIS session's store are
        // treated as synthetic.  No fallback heuristics — if the host did not
        // persist the synthetic message, the old turns are archived as fresh
        // leaves (safe: no duplicates, no broken summary chains).
        //
        // Edge case: if a user pastes an exact byte-for-byte copy of a prior
        // LCM synthetic message (including the `[LCM lcm_node=<uuid>…]`
        // header and the full summary body), it will be classified as
        // synthetic and not archived as a leaf.  This requires the user to
        // reproduce the exact UUID, header format, and summary content, which
        // is not a realistic scenario in normal usage.
        let mut real_msgs: Vec<Message> = Vec::new();
        let mut prior_summary_ids: Vec<String> = Vec::new();
        let mut max_prior_depth: u32 = 0;
        let mut prior_summary_texts: Vec<String> = Vec::new();

        for msg in &drained {
            let text = extract_text(&msg.content);
            if let Some((node_id, body)) = parse_lcm_marker(&text) {
                match self.store.get_node(&node_id) {
                    Ok(Some(node))
                        if node.session_id == session_id
                            && node.is_summary()
                            && body.starts_with(&node.content) =>
                    {
                        // Valid marker: correct session, summary node, and
                        // the body starts with the stored content.
                        prior_summary_ids.push(node.id.clone());
                        max_prior_depth = max_prior_depth.max(node.depth);
                        prior_summary_texts.push(node.content.clone());

                        // If validate_and_repair() merged a real user turn
                        // into the synthetic message, preserve the appended
                        // text as a separate real message so it's archived.
                        let tail = body[node.content.len()..].trim_start();
                        if !tail.is_empty() {
                            real_msgs.push(Message {
                                role: msg.role,
                                content: MessageContent::Text(tail.to_string()),
                            });
                        }
                    }
                    _ => {
                        // Marker present but node missing, wrong session,
                        // leaf node, or body doesn't match → real content.
                        real_msgs.push(msg.clone());
                    }
                }
            } else {
                real_msgs.push(msg.clone());
            }
        }

        if real_msgs.is_empty() {
            messages.splice(0..0, drained);
            return Ok(CompressResult::default());
        }

        let real_count = real_msgs.len();

        debug!(
            session_id = session_id,
            real = real_count,
            prior_summaries = prior_summary_ids.len(),
            "LCM: compressing messages"
        );

        // Step 1: archive real messages as leaves
        let leaf_ids = self.archive_leaves(session_id, &real_msgs)?;

        // Step 2: build summarizer input (prior context + real messages)
        let mut summarizer_input: Vec<Message> = Vec::new();
        if !prior_summary_texts.is_empty() {
            let prior_context = prior_summary_texts.join("\n");
            summarizer_input.push(Message {
                role: Role::User,
                content: MessageContent::Text(prior_context),
            });
        }
        summarizer_input.extend(real_msgs);

        // Step 3: combine children (prior summary node IDs + new leaf IDs)
        let mut child_ids = prior_summary_ids;
        child_ids.extend(leaf_ids);
        let depth = if max_prior_depth > 0 {
            max_prior_depth + 1
        } else {
            1
        };

        // Step 4: create and store summary node
        let summary = self
            .create_summary(session_id, &summarizer_input, child_ids, depth)
            .await?;

        // Step 5: update session metadata
        let total_archived = self
            .store
            .leaf_count(session_id)
            .map_err(|e| format!("lcm store error: {e}"))?;
        self.store
            .upsert_session_root(session_id, total_archived)
            .map_err(|e| format!("lcm store error: {e}"))?;

        // Step 6: inject synthetic message with embedded node ID marker
        let summary_msg = Message {
            role: Role::User,
            content: MessageContent::Text(format_lcm_summary(
                &summary.id,
                real_count,
                &summary.content,
            )),
        };
        messages.insert(0, summary_msg);

        Ok(CompressResult {
            real_count,
            summary_node_id: Some(summary.id),
        })
    }
}

/// Extract displayable text from a message's content.
///
/// Handles all `ContentBlock` variants to avoid silent data loss:
/// - `Text` → verbatim text
/// - `ToolResult` → result content
/// - `ToolUse` → `[Tool: name(input)]` representation
/// - `Thinking` → **excluded** (runtime intentionally filters chain-of-thought
///   from prompts; archiving it would re-inject reasoning traces into future
///   summaries and search results)
/// - `Image` → `[Image: media_type]` placeholder (binary data cannot be archived as text)
/// - `Unknown` → skipped
pub fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.clone()),
                ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                ContentBlock::ToolUse { name, input, .. } => {
                    Some(format!("[Tool: {name}({})]", input))
                }
                ContentBlock::Thinking { .. } => None,
                ContentBlock::Image { media_type, .. } => {
                    Some(format!("[Image: {media_type}]"))
                }
                ContentBlock::Unknown => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Marker prefix embedded in synthetic summary messages.
///
/// Format: `[LCM lcm_node=<uuid> — N earlier messages compressed]\n\n<text>`
///
/// The `lcm_node=<uuid>` tag uniquely identifies the DAG summary node.
/// This survives across turns, external drains, and serialization — no
/// in-memory index tracking required.
const LCM_MARKER_PREFIX: &str = "[LCM lcm_node=";

/// Format a synthetic summary message with the embedded node ID marker.
pub fn format_lcm_summary(node_id: &str, real_count: usize, summary_text: &str) -> String {
    format!(
        "{LCM_MARKER_PREFIX}{node_id} — {real_count} earlier messages compressed]\n\n{summary_text}"
    )
}

/// Parse the node ID from an LCM synthetic summary marker.
///
/// Validates the full header format: `[LCM lcm_node=<uuid> — N earlier messages compressed]`
/// - Must start with the exact marker prefix
/// - UUID must pass `Uuid::parse_str` (rejects malformed or pasted IDs)
/// - Must be followed by ` — ` (em-dash separator) and end with `compressed]`
///
/// Returns `None` for partial matches, quoted markers in user messages,
/// or any text that doesn't match the exact format `format_lcm_summary` produces.
/// Parse the node ID and body from an LCM synthetic summary marker.
///
/// Validates the full header: `[LCM lcm_node=<uuid> — N earlier messages compressed]\n\n<body>`
/// Returns `(node_id, body)`.  The caller must verify that `body` matches
/// the stored summary node's content to reject pasted/quoted headers.
pub fn parse_lcm_marker(text: &str) -> Option<(String, String)> {
    let rest = text.strip_prefix(LCM_MARKER_PREFIX)?;
    if rest.len() < 36 {
        return None;
    }
    let candidate = &rest[..36];
    if uuid::Uuid::parse_str(candidate).is_err() {
        return None;
    }
    let after_uuid = &rest[36..];
    if !after_uuid.starts_with(" \u{2014} ") {
        return None;
    }
    let header_close = after_uuid.find("compressed]")?;
    if header_close == 0 {
        return None;
    }
    let after_header = &after_uuid[header_close + "compressed]".len()..];
    let body = after_header.strip_prefix("\n\n").unwrap_or(after_header);
    Some((candidate.to_string(), body.to_string()))
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
            let snippet = crate::safe_truncate(&text, 200);
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

    /// Summarizer that concatenates all input message content, so tests can
    /// verify exactly what was passed to the summarizer (including prior context).
    struct VerbatimSummarizer;

    #[async_trait]
    impl LcmSummarizer for VerbatimSummarizer {
        async fn summarize(&self, messages: &[Message]) -> Result<String, String> {
            let parts: Vec<String> = messages.iter().map(|m| extract_text(&m.content)).collect();
            Ok(parts.join(" | "))
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

        let result = compressor
            .compress_oldest("sess-test", &mut messages, 2)
            .await
            .unwrap();

        assert_eq!(result.real_count, 2);
        assert!(result.summary_node_id.is_some());
        // 2 original remain + 1 injected summary = 3
        assert_eq!(messages.len(), 3);

        let first = extract_text(&messages[0].content);
        // Should have LCM marker with node ID
        assert!(parse_lcm_marker(&first).is_some());
        assert!(first.contains("2 earlier messages compressed"));
    }

    #[tokio::test]
    async fn repeated_compress_links_prior_summary() {
        let store = LcmStore::open_in_memory().unwrap();
        let compressor = LcmCompressor::new(store.clone(), Box::new(EchoSummarizer));

        let mut messages = vec![
            user_msg("a0"),
            user_msg("a1"),
            user_msg("b0"),
            user_msg("b1"),
            user_msg("c0"),
            user_msg("c1"),
        ];

        // First pass: compress 2 → leaves 0, 1
        let r1 = compressor
            .compress_oldest("sess-idx", &mut messages, 2)
            .await
            .unwrap();
        assert_eq!(r1.real_count, 2);
        // messages = [LCM_SUMMARY(node_id_1), b0, b1, c0, c1]

        // Second pass: marker-based detection finds synthetic at index 0
        let r2 = compressor
            .compress_oldest("sess-idx", &mut messages, 2)
            .await
            .unwrap();
        assert_eq!(r2.real_count, 1); // only b0 is real

        // Leaf nodes should have global indices 0, 1, 2
        let nodes = store.get_session_nodes("sess-idx").unwrap();
        let mut leaf_indices: Vec<u64> = nodes
            .iter()
            .filter(|n| n.is_leaf())
            .filter_map(|n| n.message_index)
            .collect();
        leaf_indices.sort();
        assert_eq!(leaf_indices, vec![0, 1, 2]);

        // Second summary should link prior summary (depth >= 2)
        let summaries: Vec<_> = nodes.iter().filter(|n| n.is_summary()).collect();
        let last_summary = summaries.last().unwrap();
        assert!(
            last_summary.depth >= 2,
            "Second summary should have depth >= 2, got {}",
            last_summary.depth
        );
        assert!(
            last_summary.children.len() >= 2,
            "Second summary should link prior summary + leaf, got {} children",
            last_summary.children.len()
        );
        // Prior summary ID should be in children
        let first_summary_id = &summaries[0].id;
        assert!(
            last_summary.children.contains(first_summary_id),
            "Second summary should reference first summary as child"
        );
    }

    #[tokio::test]
    async fn prior_summary_carried_into_new_summary() {
        let store = LcmStore::open_in_memory().unwrap();
        let compressor = LcmCompressor::new(store.clone(), Box::new(VerbatimSummarizer));

        let mut messages = vec![
            user_msg("alpha"),
            user_msg("beta"),
            user_msg("gamma"),
            user_msg("delta"),
        ];

        // First pass
        compressor
            .compress_oldest("sess-carry", &mut messages, 2)
            .await
            .unwrap();

        let first_summary = extract_text(&messages[0].content);
        assert!(first_summary.contains("alpha | beta"));

        // Second pass — marker detects synthetic automatically
        compressor
            .compress_oldest("sess-carry", &mut messages, 2)
            .await
            .unwrap();

        let second_summary = extract_text(&messages[0].content);
        assert!(
            second_summary.contains("alpha") && second_summary.contains("gamma"),
            "New summary should carry forward prior context. Got: {second_summary}"
        );
    }

    #[tokio::test]
    async fn compress_with_unicode_content() {
        let store = LcmStore::open_in_memory().unwrap();
        let compressor = LcmCompressor::new(store.clone(), Box::new(EchoSummarizer));

        let mut messages = vec![
            user_msg("Привіт, як справи? 🇺🇦"),
            user_msg("日本語テスト"),
        ];

        let result = compressor
            .compress_oldest("sess-unicode", &mut messages, 2)
            .await
            .unwrap();
        assert_eq!(result.real_count, 2);

        // Verify the leaf nodes were stored with correct content
        let nodes = store.get_session_nodes("sess-unicode").unwrap();
        let leaves: Vec<_> = nodes.iter().filter(|n| n.is_leaf()).collect();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].content, "Привіт, як справи? 🇺🇦");
        assert_eq!(leaves[1].content, "日本語テスト");
    }

    #[tokio::test]
    async fn fallback_summary_truncates_safely() {
        // Build a message with content > 200 bytes where byte 200 falls mid-char
        let long_cyrillic = "Б".repeat(150); // 150 × 2 bytes = 300 bytes, each char is 2 bytes
        let result = fallback_summary(&[Message {
            role: Role::User,
            content: MessageContent::Text(long_cyrillic),
        }]);
        // Should not panic and should contain the role prefix
        assert!(result.starts_with("[User]: "));
    }
}
