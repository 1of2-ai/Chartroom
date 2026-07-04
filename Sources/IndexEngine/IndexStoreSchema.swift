import Foundation

@discardableResult
private func ensureIndexStoreColumn(db: SQLite, table: String, name: String, definition: String) throws -> Bool {
    let statement = try db.prepare("PRAGMA table_info(\(table))")
    var names = Set<String>()
    while try statement.step() {
        if let columnName = statement.text(1) {
            names.insert(columnName)
        }
    }

    if !names.contains(name) {
        try db.exec("ALTER TABLE \(table) ADD COLUMN \(definition)")
        return true
    }
    return false
}

extension IndexStore {
    static func installSchema(
        db: SQLite,
        vectorBackendID: String,
        vectorBackendVersion: String
    ) throws {
        try db.exec("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          applied_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
          id TEXT PRIMARY KEY,
          connector_kind TEXT NOT NULL,
          connector_instance_id TEXT NOT NULL,
          source_uri TEXT NOT NULL,
          external_stable_id TEXT NOT NULL,
          sync_cursor TEXT NOT NULL DEFAULT '',
          capability_snapshot TEXT NOT NULL DEFAULT '{}',
          permission_scope_hash TEXT NOT NULL DEFAULT '',
          auth_reference_id TEXT NOT NULL DEFAULT '',
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          source_uri TEXT NOT NULL,
          external_id TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          version INTEGER NOT NULL,
          active_version_id TEXT NOT NULL,
          title TEXT NOT NULL,
          content_type TEXT NOT NULL,
          file_extension TEXT NOT NULL,
          size INTEGER NOT NULL,
          created_at REAL NOT NULL,
          modified_at REAL NOT NULL,
          ingested_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          is_deleted INTEGER NOT NULL DEFAULT 0,
          permission_scope_id TEXT NOT NULL DEFAULT '',
          provenance TEXT NOT NULL DEFAULT '{}',
          cluster_id TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS document_versions (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          created_at REAL NOT NULL,
          retained_state TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS representation_lineages (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          representation_kind TEXT NOT NULL,
          active_representation_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS representations (
          id TEXT PRIMARY KEY,
          lineage_id TEXT NOT NULL,
          document_id TEXT NOT NULL,
          document_version_id TEXT NOT NULL,
          representation_kind TEXT NOT NULL,
          extractor_id TEXT NOT NULL,
          extractor_version TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          language TEXT NOT NULL DEFAULT '',
          text TEXT NOT NULL,
          token_count INTEGER NOT NULL,
          content_hash TEXT NOT NULL,
          source_material_availability TEXT NOT NULL,
          upstream_dependency_state TEXT NOT NULL,
          active INTEGER NOT NULL,
          created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          document_version_id TEXT NOT NULL,
          representation_id TEXT NOT NULL,
          representation_lineage_id TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          chunker_id TEXT NOT NULL,
          chunker_version TEXT NOT NULL,
          policy_id TEXT NOT NULL,
          policy_version INTEGER NOT NULL,
          text TEXT NOT NULL,
          context_prefix TEXT NOT NULL,
          context_suffix TEXT NOT NULL,
          heading_path TEXT NOT NULL,
          byte_start INTEGER NOT NULL,
          byte_end INTEGER NOT NULL,
          character_start INTEGER NOT NULL,
          character_end INTEGER NOT NULL,
          token_start INTEGER NOT NULL,
          token_end INTEGER NOT NULL,
          page_start INTEGER NOT NULL,
          page_end INTEGER NOT NULL,
          section_label TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          active INTEGER NOT NULL,
          availability_state TEXT NOT NULL,
          created_at REAL NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
          text,
          title,
          path,
          heading_path,
          chunk_id UNINDEXED,
          tokenize='porter unicode61'
        );

        CREATE TABLE IF NOT EXISTS embeddings (
          id TEXT PRIMARY KEY,
          chunk_id TEXT NOT NULL,
          embedding_space_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          model_version TEXT NOT NULL,
          dimension INTEGER NOT NULL,
          modality TEXT NOT NULL,
          prompt_kind TEXT NOT NULL,
          vector_backend_id TEXT NOT NULL,
          vector_backend_version TEXT NOT NULL,
          vector_hash TEXT NOT NULL,
          created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vectors (
          id TEXT PRIMARY KEY,
          dim INTEGER NOT NULL,
          vec BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vector_backend_metadata (
          id TEXT PRIMARY KEY,
          version TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policies (
          id TEXT PRIMARY KEY,
          version INTEGER NOT NULL,
          raw_retention TEXT NOT NULL,
          extractor_id TEXT NOT NULL,
          chunker_id TEXT NOT NULL,
          embedding_provider_id TEXT NOT NULL,
          vector_backend_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policy_component_resolutions (
          policy_id TEXT NOT NULL,
          component_id TEXT NOT NULL,
          component_kind TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          PRIMARY KEY(policy_id, component_id, component_kind)
        );

        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY,
          state TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'ingest',
          completed_unit_count INTEGER NOT NULL,
          total_unit_count INTEGER,
          message TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS failures (
          id TEXT PRIMARY KEY,
          category TEXT NOT NULL,
          message TEXT NOT NULL,
          detail TEXT NOT NULL,
          source_id TEXT,
          document_id TEXT,
          source_uri TEXT,
          is_recoverable INTEGER NOT NULL,
          recoverability TEXT NOT NULL DEFAULT 'retryable',
          occurred_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS search_diagnostics (
          id TEXT PRIMARY KEY,
          query TEXT NOT NULL,
          diagnostics_json TEXT NOT NULL,
          created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS objects (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          title TEXT NOT NULL,
          cluster_id TEXT,
          source_id TEXT,
          source_uri TEXT,
          policy_id TEXT,
          representation_id TEXT,
          embedding_space_id TEXT,
          model_id TEXT NOT NULL,
          updated_at REAL NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(
          body,
          id UNINDEXED,
          tokenize='porter unicode61'
        );

        CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
        CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(content_type);
        CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active);
        CREATE INDEX IF NOT EXISTS idx_embeddings_space ON embeddings(embedding_space_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_failures_time ON failures(occurred_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_time ON jobs(updated_at);
        """)
        try ensureIndexStoreColumn(db: db, table: "objects", name: "source_id", definition: "source_id TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "source_uri", definition: "source_uri TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "policy_id", definition: "policy_id TEXT")
        try ensureIndexStoreColumn(db: db, table: "objects", name: "representation_id", definition: "representation_id TEXT")
        // Tables that never earned readers (#66): declared in early schemas, written
        // never (or only with placeholder values). Dropped from existing stores too.
        try db.exec("""
        DROP TABLE IF EXISTS source_cursors;
        DROP TABLE IF EXISTS raw_blobs;
        DROP TABLE IF EXISTS relations;
        DROP TABLE IF EXISTS chunk_metadata;
        """)
        try ensureIndexStoreColumn(db: db, table: "objects", name: "embedding_space_id", definition: "embedding_space_id TEXT")
        try ensureIndexStoreColumn(db: db, table: "jobs", name: "kind", definition: "kind TEXT NOT NULL DEFAULT 'ingest'")
        try ensureIndexStoreColumn(db: db, table: "failures", name: "source_uri", definition: "source_uri TEXT")
        let addedRecoverabilityColumn = try ensureIndexStoreColumn(
            db: db,
            table: "failures",
            name: "recoverability",
            definition: "recoverability TEXT NOT NULL DEFAULT 'retryable'"
        )
        if addedRecoverabilityColumn {
            try db.exec("""
            UPDATE failures
            SET recoverability = CASE WHEN is_recoverable != 0 THEN 'retryable' ELSE 'unrecoverable' END;
            """)
        } else {
            try db.exec("""
            UPDATE failures
            SET recoverability = CASE WHEN is_recoverable != 0 THEN 'retryable' ELSE 'unrecoverable' END
            WHERE recoverability IS NULL OR recoverability = '';
            """)
        }
        try db.exec("""
        CREATE INDEX IF NOT EXISTS idx_objects_cluster ON objects(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_objects_source ON objects(source_id);
        CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(type);
        CREATE INDEX IF NOT EXISTS idx_objects_policy ON objects(policy_id);
        CREATE INDEX IF NOT EXISTS idx_objects_embedding_space ON objects(embedding_space_id);
        """)

        let migration = try db.prepare("""
        INSERT OR IGNORE INTO schema_migrations(version,name,applied_at)
        VALUES(1,'durable-retrieval-core',?1)
        """)
        migration.bind(1, Date.now.timeIntervalSince1970)
        try migration.step()

        let backend = try db.prepare("""
        INSERT INTO vector_backend_metadata(id,version,state,message,updated_at)
        VALUES(?1,?2,?3,?4,?5)
        ON CONFLICT(id) DO UPDATE SET version=excluded.version,
          state=excluded.state,
          message=excluded.message,
          updated_at=excluded.updated_at
        """)
        backend.bind(1, vectorBackendID)
        backend.bind(2, vectorBackendVersion)
        backend.bind(3, VectorStorageStatus.State.ready.rawValue)
        backend.bind(4, "Exact vector scan over SQLite-backed chunk embeddings")
        backend.bind(5, Date.now.timeIntervalSince1970)
        try backend.step()
    }
}
