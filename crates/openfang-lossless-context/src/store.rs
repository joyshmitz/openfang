//! SQLite-backed persistence for LCM DAG nodes.
//!
//! Two tables:
//! - `lcm_nodes`         — individual DAG nodes (leaves and summaries)
//! - `lcm_session_roots` — per-session metadata (message count, last compression time)

use rusqlite::{params, Connection, Result};
use std::sync::{Arc, Mutex};

use crate::dag::{DagNode, NodeType};

/// SQLite store for LCM DAG nodes.
///
/// Cheaply cloneable — clone shares the same underlying SQLite connection
/// via the inner `Arc<Mutex<Connection>>`.
pub struct LcmStore {
    conn: Arc<Mutex<Connection>>,
}

impl Clone for LcmStore {
    fn clone(&self) -> Self {
        Self {
            conn: Arc::clone(&self.conn),
        }
    }
}

impl LcmStore {
    /// Open (or create) the LCM store at the given SQLite database path.
    pub fn open(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.migrate()?;
        Ok(store)
    }

    /// Open an in-memory store (primarily for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.migrate()?;
        Ok(store)
    }

    /// Apply the LCM schema migrations.
    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS lcm_nodes (
                id             TEXT    PRIMARY KEY,
                session_id     TEXT    NOT NULL,
                parent_id      TEXT,
                node_type      TEXT    NOT NULL,
                role           TEXT,
                content        TEXT    NOT NULL,
                message_index  INTEGER,
                children       TEXT    NOT NULL DEFAULT '[]',
                token_estimate INTEGER NOT NULL DEFAULT 0,
                created_at     INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_lcm_nodes_session
                ON lcm_nodes(session_id);

            CREATE INDEX IF NOT EXISTS idx_lcm_nodes_type
                ON lcm_nodes(session_id, node_type);

            CREATE TABLE IF NOT EXISTS lcm_session_roots (
                session_id      TEXT    PRIMARY KEY,
                total_messages  INTEGER NOT NULL DEFAULT 0,
                compressed_at   INTEGER NOT NULL
            );
            "#,
        )?;
        Ok(())
    }

    /// Insert or replace a DAG node.
    pub fn insert_node(&self, node: &DagNode) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let node_type = node.node_type.to_string();
        let children_json =
            serde_json::to_string(&node.children).unwrap_or_else(|_| "[]".to_string());
        conn.execute(
            "INSERT OR REPLACE INTO lcm_nodes
             (id, session_id, parent_id, node_type, role, content,
              message_index, children, token_estimate, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                node.id,
                node.session_id,
                node.parent_id,
                node_type,
                node.role,
                node.content,
                node.message_index.map(|i| i as i64),
                children_json,
                node.token_estimate as i64,
                node.created_at,
            ],
        )?;
        Ok(())
    }

    /// Retrieve a single node by ID.
    pub fn get_node(&self, id: &str) -> Result<Option<DagNode>> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, session_id, parent_id, node_type, role, content,
                    message_index, children, token_estimate, created_at
             FROM lcm_nodes WHERE id = ?1",
        )?;
        let node = stmt
            .query_row(params![id], parse_row)
            .ok();
        Ok(node)
    }

    /// Retrieve all nodes for a session, ordered by original message index then creation time.
    pub fn get_session_nodes(&self, session_id: &str) -> Result<Vec<DagNode>> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, session_id, parent_id, node_type, role, content,
                    message_index, children, token_estimate, created_at
             FROM lcm_nodes
             WHERE session_id = ?1
             ORDER BY COALESCE(message_index, 999999), created_at",
        )?;
        let nodes = stmt
            .query_map(params![session_id], parse_row)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(nodes)
    }

    /// Full-text search across all nodes for a session (LIKE matching).
    pub fn search_nodes(&self, session_id: &str, query: &str) -> Result<Vec<DagNode>> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let pattern = format!("%{}%", query);
        let mut stmt = conn.prepare(
            "SELECT id, session_id, parent_id, node_type, role, content,
                    message_index, children, token_estimate, created_at
             FROM lcm_nodes
             WHERE session_id = ?1 AND content LIKE ?2
             ORDER BY created_at DESC
             LIMIT 50",
        )?;
        let nodes = stmt
            .query_map(params![session_id, pattern], parse_row)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(nodes)
    }

    /// Update (or insert) the session root metadata.
    pub fn upsert_session_root(&self, session_id: &str, total_messages: u64) -> Result<()> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO lcm_session_roots
             (session_id, total_messages, compressed_at)
             VALUES (?1, ?2, ?3)",
            params![session_id, total_messages as i64, now],
        )?;
        Ok(())
    }

    /// Return the total number of original messages tracked for a session.
    pub fn session_message_count(&self, session_id: &str) -> Result<u64> {
        let conn = self.conn.lock().expect("lcm store mutex poisoned");
        let count: i64 = conn
            .query_row(
                "SELECT COALESCE(total_messages, 0) FROM lcm_session_roots
                 WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(count as u64)
    }
}

/// Parse a SQLite row into a `DagNode`.
fn parse_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DagNode> {
    let id: String = row.get(0)?;
    let session_id: String = row.get(1)?;
    let parent_id: Option<String> = row.get(2)?;
    let node_type_str: String = row.get(3)?;
    let role: Option<String> = row.get(4)?;
    let content: String = row.get(5)?;
    let message_index: Option<i64> = row.get(6)?;
    let children_json: String = row.get(7)?;
    let token_estimate: i64 = row.get(8)?;
    let created_at: i64 = row.get(9)?;

    let node_type = if node_type_str == "summary" {
        NodeType::Summary
    } else {
        NodeType::Leaf
    };
    let children: Vec<String> =
        serde_json::from_str(&children_json).unwrap_or_default();

    Ok(DagNode {
        id,
        session_id,
        parent_id,
        node_type,
        role,
        content,
        message_index: message_index.map(|i| i as u64),
        children,
        token_estimate: token_estimate as u64,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DagNode;

    #[test]
    fn roundtrip_leaf_node() {
        let store = LcmStore::open_in_memory().unwrap();
        let node = DagNode::new_leaf("sess-1", "user", "Hello world", 0);
        store.insert_node(&node).unwrap();

        let fetched = store.get_node(&node.id).unwrap().unwrap();
        assert_eq!(fetched.id, node.id);
        assert_eq!(fetched.content, "Hello world");
        assert!(fetched.is_leaf());
    }

    #[test]
    fn search_finds_matching_content() {
        let store = LcmStore::open_in_memory().unwrap();
        let node = DagNode::new_leaf("sess-2", "user", "Rust is great for systems programming", 0);
        store.insert_node(&node).unwrap();

        let results = store.search_nodes("sess-2", "systems").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, node.id);
    }

    #[test]
    fn session_root_upsert() {
        let store = LcmStore::open_in_memory().unwrap();
        store.upsert_session_root("sess-3", 42).unwrap();
        assert_eq!(store.session_message_count("sess-3").unwrap(), 42);

        // Update
        store.upsert_session_root("sess-3", 100).unwrap();
        assert_eq!(store.session_message_count("sess-3").unwrap(), 100);
    }
}
