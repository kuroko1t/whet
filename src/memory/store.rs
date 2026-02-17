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
            std::fs::create_dir_all(parent).ok();
        }

        let conn = Connection::open(&expanded)?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    /// Create an in-memory store (for testing).
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
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );",
        )?;
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
    ) -> SqliteResult<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_call_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![conversation_id, role, content, tool_call_id, now],
        )?;
        // Update conversation timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conversation_id],
        )?;
        Ok(())
    }

    pub fn load_messages(
        &self,
        conversation_id: &str,
    ) -> SqliteResult<Vec<(String, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content, tool_call_id FROM messages
             WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        rows.collect()
    }

    pub fn get_latest_conversation_id(&self) -> SqliteResult<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1")?;
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
            .save_message("test-conv-1", "user", "Hello", None)
            .unwrap();
        store
            .save_message("test-conv-1", "assistant", "Hi there!", None)
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

        store.create_conversation("conv-1").unwrap();
        store.create_conversation("conv-2").unwrap();

        let latest = store.get_latest_conversation_id().unwrap();
        assert!(latest.is_some());
    }

    #[test]
    fn test_tool_call_id_stored() {
        let store = MemoryStore::in_memory().unwrap();
        store.create_conversation("conv-tool").unwrap();
        store
            .save_message("conv-tool", "tool", "file contents here", Some("call_0"))
            .unwrap();

        let messages = store.load_messages("conv-tool").unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].2, Some("call_0".to_string()));
    }
}
