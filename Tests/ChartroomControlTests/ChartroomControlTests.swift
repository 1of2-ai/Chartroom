import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing
@testable import ChartroomControl

@Suite("Chartroom session")
struct ChartroomSessionTests {
    @Test("opens an engine and reports status through the product command surface")
    func opensAndReportsStatus() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )

        let status = try await session.open()

        #expect(status.state == .ready)
        #expect(status.snapshot?.documentCount == 0)
        #expect(status.health?.documentCount == 0)
        #expect(status.modelStatus?.isAvailable == true)
    }

    @Test("ingests a local source, searches it, browses it, inspects chunks, and deletes it")
    func localSourceFeatureParityCommands() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-session-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let noteURL = root.appending(path: "Needle.md")
        try "headless mcp parity search needle".write(to: noteURL, atomically: true, encoding: .utf8)

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        _ = try await session.open()

        let sync = try await session.ingestLocalSource(root)
        #expect(sync.accepted == 1)
        #expect(sync.totalFailed == 0)

        let search = try await session.search(.init(query: "parity needle", mode: .diagnostic, limit: 5))
        let result = try #require(search.results.first)
        #expect(result.documentID == "local-files:Needle.md")

        let browse = try await session.browseDocuments(.init(limit: 10))
        #expect(browse.documents.map(\.id) == ["local-files:Needle.md"])

        let chunks = try await session.chunks(forDocument: result.documentID)
        #expect(chunks.count == 1)
        #expect(chunks.first?.text.contains("headless mcp parity") == true)

        let deletion = try await session.deleteDocuments([result.documentID])
        #expect(deletion.deletedCount == 1)

        let empty = try await session.search(.init(query: "parity needle", mode: .diagnostic, limit: 5))
        #expect(empty.results.isEmpty)
    }
}

private final class InMemoryCursorStore: CursorStore, @unchecked Sendable {
    private let lock = NSLock()
    private var cursors: [String: SourceCursor] = [:]

    func cursor(forKey key: String) -> SourceCursor? {
        lock.withLock { cursors[key] }
    }

    func setCursor(_ cursor: SourceCursor, forKey key: String) {
        lock.withLock {
            cursors[key] = cursor
        }
    }
}
