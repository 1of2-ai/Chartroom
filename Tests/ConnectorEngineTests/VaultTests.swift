import Foundation
import Testing
import IndexEngine
@testable import ConnectorEngine

@Suite("Vault ingestion (Obsidian markdown)")
struct VaultTests {
    @Test("parses frontmatter and body")
    func parsesFrontmatter() {
        let md = """
        ---
        id: n-capture
        type: note
        title: Capture Plan
        cluster: c1
        ---
        # Heading
        thermal rig notes here
        """
        let o = Vault.parse(markdown: md, relativePath: "Projects/Capture.md")
        #expect(o.id == "n-capture")
        #expect(o.type == "note")
        #expect(o.title == "Capture Plan")
        #expect(o.clusterID == "c1")
        #expect(o.body.contains("thermal rig notes"))
    }

    @Test("derives defaults without frontmatter")
    func derivesDefaults() {
        let o = Vault.parse(markdown: "# Budget Ask\nsitting with finance", relativePath: "Inbox/budget.md")
        #expect(o.id == "Inbox/budget.md")
        #expect(o.type == "note")
        #expect(o.title == "Budget Ask")
        #expect(o.clusterID == nil)
    }

    @Test("falls back to the file stem when there is no heading")
    func fileStemTitle() {
        let o = Vault.parse(markdown: "just some body text", relativePath: "Notes/thermal-pad.md")
        #expect(o.title == "thermal-pad")
    }

    @Test("treats unclosed frontmatter as body text")
    func unclosedFrontmatterIsBodyText() {
        let o = Vault.parse(markdown: "---\ntitle: Not Frontmatter\n# Actual Heading\nbody", relativePath: "Notes/broken.md")
        #expect(o.id == "Notes/broken.md")
        #expect(o.title == "Actual Heading")
        #expect(o.body.contains("title: Not Frontmatter"))
    }

    @Test("ingests a directory of markdown and it becomes searchable")
    func ingestsAndSearches() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vault-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "---\ntitle: Thermal\n---\nthe m-series rig is thermal throttling"
            .write(to: dir.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try "# Budget\nthe studio budget ask is with finance"
            .write(to: dir.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 256))
        let n = try await store.ingestVault(at: dir)
        #expect(n == 2)
        #expect(try await store.count() == 2)

        let hits = try await store.search("thermal throttling", limit: 5)
        #expect(hits.first?.title == "Thermal")
    }

    @Test("bad vault files record failures without aborting the walk")
    func badVaultFilesRecordFailuresWithoutAborting() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vault-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "# Good\nsearchable thermal note"
            .write(to: dir.appendingPathComponent("good.md"), atomically: true, encoding: .utf8)
        try Data([0xFF, 0xFE, 0x00])
            .write(to: dir.appendingPathComponent("bad.md"))

        let store = try IndexStore(path: ":memory:", embedder: HashingEmbedder(dimension: 256))
        let count = try await store.ingestVault(at: dir)
        #expect(count == 1)
        #expect(try await store.count() == 1)

        let failures = try await store.failureSnapshots(limit: 10)
        let badFile = try #require(failures.first { $0.documentID == "bad.md" })
        #expect(badFile.category == .extractionFailure)
        #expect(badFile.recoverability == .needsUserAction)

        let hits = try await store.search("thermal", limit: 5)
        #expect(hits.first?.documentID == "good.md")
    }
}
