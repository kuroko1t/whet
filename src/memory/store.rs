use rusqlite::{params, Connection, Result as SqliteResult};

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
        // Migration: add tool_calls column for existing databases
        let _ = self
            .conn
            .execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT", []);
        Ok(())
    }

    pub fn create_conversation(&self, id: &str) -> SqliteResult<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO conversations (id, created_at, updated_at) VALUES (?1, ?2, ?3)",
            params![id, now, now],
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

    pub fn get_latest_conversation_id(&self) -> SqliteResult<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id FROM conversations c
             WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)
             ORDER BY c.updated_at DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_load_conversation() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("test-conv-1").unwrap();
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
        assert!(store.get_latest_conversation_id().unwrap().is_none());

        // Empty conversations should not be returned
        store.create_conversation("conv-1").unwrap();
        store.create_conversation("conv-2").unwrap();
        assert!(store.get_latest_conversation_id().unwrap().is_none());

        // Only conversations with messages should be returned
        store
            .save_message("conv-1", "user", "hello", None, None)
            .unwrap();
        let latest = store.get_latest_conversation_id().unwrap();
        assert_eq!(latest, Some("conv-1".to_string()));
    }

    #[test]
    fn test_tool_call_id_stored() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-tool").unwrap();
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
        store.create_conversation("empty-conv").unwrap();
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
        store.create_conversation("dup-id").unwrap();
        let result = store.create_conversation("dup-id");
        assert!(result.is_err());
    }

    #[test]
    fn test_message_ordering_preserved() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("order-test").unwrap();
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
        store.create_conversation("large-test").unwrap();
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
        store.create_conversation("special-chars").unwrap();
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
        store.create_conversation("conv-a").unwrap();
        store.create_conversation("conv-b").unwrap();

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
        store.create_conversation("older").unwrap();
        store.create_conversation("newer").unwrap();

        // Save message to "older" to make it the most recently updated
        store
            .save_message("older", "user", "update", None, None)
            .unwrap();

        let latest = store.get_latest_conversation_id().unwrap().unwrap();
        assert_eq!(latest, "older");
    }

    #[test]
    fn test_tool_calls_json_stored_and_loaded() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-tc").unwrap();

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
        store.create_conversation("conv-bc").unwrap();
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
            store.create_conversation("persist-test").unwrap();
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
}
