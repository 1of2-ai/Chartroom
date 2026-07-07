import ChartroomControl
import ChartroomTestSupport
import Foundation
import IndexEngine
import Testing

@Suite("ChartroomControl fixture corpus")
struct ChartroomControlFixtureCorpusTests {
    @Test("shared filesystem fixture drives the session command surface")
    func sharedFilesystemFixtureDrivesSessionCommands() async throws {
        let workspace = try FixtureCorpora.writeBasicNotes()
        defer { workspace.cleanup() }

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: FixtureCursorStore()
        )
        _ = try await session.open()

        let sync = try await session.ingestLocalSource(workspace.rootURL)
        #expect(SyncOutcomeProjection(sync).accepted == 4)

        let search = try await session.search(.init(query: "atlas routing needle", mode: .diagnostic, limit: 5))
        #expect(SearchProjection(search).documentIDs.first == "local-files:Atlas.md")

        let browse = try await session.browseDocuments(.init(sort: .titleAscending, limit: 10))
        let browseProjection = BrowseProjection(browse)
        #expect(browseProjection.documentIDs == [
            "local-files:Atlas.md",
            "local-files:Beacon.md",
            "local-files:Research/Compass.md",
            "local-files:Unicode.md",
        ])

        let chunks = try await session.chunks(forDocument: "local-files:Atlas.md")
        #expect(ChunkProjection(chunks).texts.first?.contains("stable search projections") == true)
    }

    @Test("mutation fixture updates and deletes documents across syncs")
    func mutationFixtureUpdatesAndDeletesDocumentsAcrossSyncs() async throws {
        let workspace = try FixtureCorpora.writeMutationSeed()
        defer { workspace.cleanup() }

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: FixtureCursorStore()
        )
        _ = try await session.open()

        let firstSync = try await session.ingestLocalSource(workspace.rootURL)
        #expect(SyncOutcomeProjection(firstSync).accepted == 2)

        try workspace.write("Stable.md", contents: "Stable edited mutation needle remains searchable.")
        try workspace.remove("Removed.md")

        let secondSync = try await session.ingestLocalSource(workspace.rootURL)
        let secondProjection = SyncOutcomeProjection(secondSync)
        #expect(secondProjection.accepted == 1)
        #expect(secondProjection.deletedCount == 1)
        #expect(secondProjection.totalFailed == 0)

        let edited = try await session.search(.init(query: "edited mutation needle", mode: .diagnostic, limit: 5))
        #expect(SearchProjection(edited).documentIDs == ["local-files:Stable.md"])

        let browse = try await session.browseDocuments(.init(limit: 10))
        #expect(!BrowseProjection(browse).documentIDs.contains("local-files:Removed.md"))
    }
}
