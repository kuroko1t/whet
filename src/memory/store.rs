use rusqlite::{params, Connection, Result as SqliteResult};

pub struct ConversationSummary {
    pub id: String,
    pub title: Option<String>,
    pub updated_at: String,
    pub message_count: usize,
}

pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    pub fn new(db_path: &str) -> SqliteResult<Self> {
        // Expand ~ to home directory
        let expanded = if db_path.starts_with('~') {
            if let Some(home) = dirs::home_dir() {
                db_path.replacen('~', &home.display().to_string(), 1)
            } else {
                db_path.to_string()
            }
        } else {
            db_path.to_string()
        };

        // Create parent directory if needed
        if let Some(parent) = std::path::Path::new(&expanded).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "Warning: Failed to create directory {}: {}",
                    parent.display(),
                    e
                );
            }
        }

        let conn = Connection::open(&expanded)?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    /// Create an in-memory store (for testing).
    #[allow(dead_code)]
    pub fn in_memory() -> SqliteResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    fn init_tables(&self) -> SqliteResult<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                working_dir TEXT,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_call_id TEXT,
                tool_calls TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );",
        )?;
        // Migrations for existing databases
        let _ = self
            .conn
            .execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE conversations ADD COLUMN working_dir TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE conversations ADD COLUMN title TEXT", []);
        Ok(())
    }

    pub fn create_conversation(&self, id: &str, working_dir: &str) -> SqliteResult<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO conversations (id, working_dir, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, working_dir, now, now],
        )?;
        Ok(())
    }

    pub fn save_message(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
        tool_call_id: Option<&str>,
        tool_calls_json: Option<&str>,
    ) -> SqliteResult<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_call_id, tool_calls, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conversation_id, role, content, tool_call_id, tool_calls_json, now],
        )?;
        // Update conversation timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conversation_id],
        )?;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn load_messages(
        &self,
        conversation_id: &str,
    ) -> SqliteResult<Vec<(String, String, Option<String>, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content, tool_call_id, tool_calls FROM messages
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?;
        rows.collect()
    }

    pub fn get_latest_conversation_id(&self, working_dir: &str) -> SqliteResult<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM conversations c
             WHERE (c.working_dir = ?1 OR c.working_dir IS NULL)
               AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
             ORDER BY c.updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query(params![working_dir])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn update_conversation_title(&self, id: &str, title: &str) -> SqliteResult<()> {
        self.conn.execute(
            "UPDATE conversations SET title = ?1 WHERE id = ?2",
            params![title, id],
        )?;
        Ok(())
    }

    pub fn list_conversations(&self, working_dir: &str) -> SqliteResult<Vec<ConversationSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.title, c.updated_at, COUNT(m.id) as msg_count
             FROM conversations c
             JOIN messages m ON m.conversation_id = c.id
             WHERE c.working_dir = ?1 OR c.working_dir IS NULL
             GROUP BY c.id
             HAVING msg_count > 0
             ORDER BY c.updated_at DESC",
        )?;
        let rows = stmt.query_map(params![working_dir], |row| {
            Ok(ConversationSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                updated_at: row.get(2)?,
                message_count: row.get::<_, i64>(3)? as usize,
            })
        })?;
        rows.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_load_conversation() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("test-conv-1", "/tmp").unwrap();
        store
            .save_message("test-conv-1", "user", "Hello", None, None)
            .unwrap();
        store
            .save_message("test-conv-1", "assistant", "Hi there!", None, None)
            .unwrap();

        let messages = store.load_messages("test-conv-1").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].0, "user");
        assert_eq!(messages[0].1, "Hello");
        assert_eq!(messages[1].0, "assistant");
        assert_eq!(messages[1].1, "Hi there!");
    }

    #[test]
    fn test_get_latest_conversation() {
        let store = MemoryStore::in_memory().unwrap();
        assert!(store.get_latest_conversation_id("/tmp").unwrap().is_none());

        // Empty conversations should not be returned
        store.create_conversation("conv-1", "/tmp").unwrap();
        store.create_conversation("conv-2", "/tmp").unwrap();
        assert!(store.get_latest_conversation_id("/tmp").unwrap().is_none());

        // Only conversations with messages should be returned
        store
            .save_message("conv-1", "user", "hello", None, None)
            .unwrap();
        let latest = store.get_latest_conversation_id("/tmp").unwrap();
        assert_eq!(latest, Some("conv-1".to_string()));
    }

    #[test]
    fn test_get_latest_conversation_filters_by_dir() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-a", "/project-a").unwrap();
        store.create_conversation("conv-b", "/project-b").unwrap();
        store
            .save_message("conv-a", "user", "hello from A", None, None)
            .unwrap();
        store
            .save_message("conv-b", "user", "hello from B", None, None)
            .unwrap();

        let latest_a = store.get_latest_conversation_id("/project-a").unwrap();
        assert_eq!(latest_a, Some("conv-a".to_string()));

        let latest_b = store.get_latest_conversation_id("/project-b").unwrap();
        assert_eq!(latest_b, Some("conv-b".to_string()));

        // No conversations for unknown dir
        assert!(store
            .get_latest_conversation_id("/unknown")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_tool_call_id_stored() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-tool", "/tmp").unwrap();
        store
            .save_message(
                "conv-tool",
                "tool",
                "file contents here",
                Some("call_0"),
                None,
            )
            .unwrap();

        let messages = store.load_messages("conv-tool").unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].2, Some("call_0".to_string()));
    }

    #[test]
    fn test_load_empty_conversation() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("empty-conv", "/tmp").unwrap();
        let messages = store.load_messages("empty-conv").unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_load_nonexistent_conversation() {
        let store = MemoryStore::in_memory().unwrap();
        let messages = store.load_messages("nonexistent").unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_duplicate_conversation_id_fails() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("dup-id", "/tmp").unwrap();
        let result = store.create_conversation("dup-id", "/tmp");
        assert!(result.is_err());
    }

    #[test]
    fn test_message_ordering_preserved() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("order-test", "/tmp").unwrap();
        for i in 0..10 {
            store
                .save_message("order-test", "user", &format!("msg_{}", i), None, None)
                .unwrap();
        }
        let messages = store.load_messages("order-test").unwrap();
        assert_eq!(messages.len(), 10);
        for (i, msg) in messages.iter().enumerate() {
            assert_eq!(msg.1, format!("msg_{}", i));
        }
    }

    #[test]
    fn test_large_content_stored() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("large-test", "/tmp").unwrap();
        let large_content = "x".repeat(100_000);
        store
            .save_message("large-test", "user", &large_content, None, None)
            .unwrap();

        let messages = store.load_messages("large-test").unwrap();
        assert_eq!(messages[0].1.len(), 100_000);
    }

    #[test]
    fn test_special_characters_in_content() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("special-chars", "/tmp").unwrap();
        let content = "Hello 'world' \"test\" \\ \n\t\r\0 日本語 🦀";
        store
            .save_message("special-chars", "user", content, None, None)
            .unwrap();

        let messages = store.load_messages("special-chars").unwrap();
        assert_eq!(messages[0].1, content);
    }

    #[test]
    fn test_multiple_conversations_isolated() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-a", "/tmp").unwrap();
        store.create_conversation("conv-b", "/tmp").unwrap();

        store
            .save_message("conv-a", "user", "message for A", None, None)
            .unwrap();
        store
            .save_message("conv-b", "user", "message for B", None, None)
            .unwrap();

        let messages_a = store.load_messages("conv-a").unwrap();
        let messages_b = store.load_messages("conv-b").unwrap();
        assert_eq!(messages_a.len(), 1);
        assert_eq!(messages_b.len(), 1);
        assert_eq!(messages_a[0].1, "message for A");
        assert_eq!(messages_b[0].1, "message for B");
    }

    #[test]
    fn test_latest_conversation_updated_on_message() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("older", "/tmp").unwrap();
        store.create_conversation("newer", "/tmp").unwrap();

        // Save message to "older" to make it the most recently updated
        store
            .save_message("older", "user", "update", None, None)
            .unwrap();

        let latest = store.get_latest_conversation_id("/tmp").unwrap().unwrap();
        assert_eq!(latest, "older");
    }

    #[test]
    fn test_tool_calls_json_stored_and_loaded() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-tc", "/tmp").unwrap();

        let tc_json = r#"[{"id":"call_0","name":"read_file","arguments":{"path":"src/main.rs"}}]"#;
        store
            .save_message("conv-tc", "assistant", "", None, Some(tc_json))
            .unwrap();
        store
            .save_message("conv-tc", "tool", "file contents", Some("call_0"), None)
            .unwrap();

        let messages = store.load_messages("conv-tc").unwrap();
        assert_eq!(messages.len(), 2);
        // assistant message with tool_calls
        assert_eq!(messages[0].0, "assistant");
        assert_eq!(messages[0].3, Some(tc_json.to_string()));
        // tool result
        assert_eq!(messages[1].0, "tool");
        assert_eq!(messages[1].2, Some("call_0".to_string()));
        assert!(messages[1].3.is_none());
    }

    #[test]
    fn test_tool_calls_null_backward_compat() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-bc", "/tmp").unwrap();
        // Save without tool_calls (simulating old DB rows)
        store
            .save_message("conv-bc", "assistant", "hello", None, None)
            .unwrap();

        let messages = store.load_messages("conv-bc").unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].3.is_none());
    }

    #[test]
    fn test_file_based_store() {
        let path = "/tmp/whet_test_memory.db";
        // Clean up from previous runs
        std::fs::remove_file(path).ok();

        {
            let store = MemoryStore::new(path).unwrap();
            store.create_conversation("persist-test", "/tmp").unwrap();
            store
                .save_message("persist-test", "user", "persistent message", None, None)
                .unwrap();
        }

        // Re-open and verify data persists
        {
            let store = MemoryStore::new(path).unwrap();
            let messages = store.load_messages("persist-test").unwrap();
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].1, "persistent message");
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_list_conversations() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-1", "/project").unwrap();
        store.create_conversation("conv-2", "/project").unwrap();
        store.create_conversation("conv-3", "/other").unwrap();

        // conv-1: has messages
        store
            .save_message("conv-1", "user", "hello", None, None)
            .unwrap();
        store
            .save_message("conv-1", "assistant", "hi", None, None)
            .unwrap();
        // conv-2: has messages
        store
            .save_message("conv-2", "user", "world", None, None)
            .unwrap();
        // conv-3: different dir, has messages
        store
            .save_message("conv-3", "user", "other dir", None, None)
            .unwrap();

        let convs = store.list_conversations("/project").unwrap();
        assert_eq!(convs.len(), 2);
        // Most recent first
        assert_eq!(convs[0].id, "conv-2");
        assert_eq!(convs[0].message_count, 1);
        assert_eq!(convs[1].id, "conv-1");
        assert_eq!(convs[1].message_count, 2);

        // Different directory
        let other_convs = store.list_conversations("/other").unwrap();
        assert_eq!(other_convs.len(), 1);
        assert_eq!(other_convs[0].id, "conv-3");
    }

    #[test]
    fn test_list_conversations_empty_dir() {
        let store = MemoryStore::in_memory().unwrap();
        let convs = store.list_conversations("/empty").unwrap();
        assert!(convs.is_empty());
    }

    #[test]
    fn test_update_conversation_title() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-t", "/tmp").unwrap();
        store
            .save_message("conv-t", "user", "hello", None, None)
            .unwrap();

        store
            .update_conversation_title("conv-t", "fix the bug in login")
            .unwrap();

        let convs = store.list_conversations("/tmp").unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].title, Some("fix the bug in login".to_string()));
    }

    #[test]
    fn test_backward_compat_null_working_dir() {
        let store = MemoryStore::in_memory().unwrap();
        // Simulate old data with NULL working_dir
        let now = chrono::Utc::now().to_rfc3339();
        store
            .conn
            .execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?1, ?2, ?3)",
                params!["old-conv", now, now],
            )
            .unwrap();
        store
            .save_message("old-conv", "user", "old message", None, None)
            .unwrap();

        // Should appear in any working_dir listing (backward compat)
        let convs = store.list_conversations("/any-dir").unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].id, "old-conv");

        // Should also be found by get_latest
        let latest = store.get_latest_conversation_id("/any-dir").unwrap();
        assert_eq!(latest, Some("old-conv".to_string()));
    }
}
