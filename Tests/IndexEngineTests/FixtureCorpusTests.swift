import ChartroomTestSupport
import Foundation
import IndexEngine
import Testing

@Suite("IndexEngine fixture corpus")
struct IndexEngineFixtureCorpusTests {
    @Test("direct payload fixtures ingest, search, browse, and inspect chunks")
    func directPayloadFixturesExerciseLibraryContract() async throws {
        let engine = try await IndexEngine.openInMemory()
        let payloads = FixturePayloads.basicNotes()

        let ingest = try await engine.ingest(.init(payloads: payloads, jobID: "fixture-basic-ingest"))
        #expect(ingest.acceptedCount == 3)
        #expect(ingest.failedCount == 0)

        let search = try await engine.search(.init(query: "atlas routing needle", mode: .diagnostic, limit: 5))
        let searchProjection = SearchProjection(search)
        #expect(searchProjection.documentIDs.first == "fixture:atlas")
        #expect(searchProjection.snippets.first?.contains("Atlas routing needle") == true)

        let browse = try await engine.browseDocuments(.init(
            filters: .init(sourceIDs: ["fixture-source"]),
            sort: .titleAscending,
            limit: 10
        ))
        let browseProjection = BrowseProjection(browse)
        #expect(browseProjection.documentIDs == ["fixture:atlas", "fixture:beacon", "fixture:compass"])
        #expect(browseProjection.titles == ["Atlas", "Beacon", "Compass"])

        let chunks = try await engine.chunks(forDocument: "fixture:atlas")
        let chunkProjection = ChunkProjection(chunks)
        #expect(chunkProjection.documentIDs == ["fixture:atlas"])
        #expect(chunkProjection.texts.first?.contains("source filters") == true)
    }
}
