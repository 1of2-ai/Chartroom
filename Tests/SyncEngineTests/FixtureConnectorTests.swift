import ChartroomTestSupport
import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing

@Suite("SyncEngine fixture connector")
struct SyncEngineFixtureConnectorTests {
    @Test("scripted connector fixtures drive the real engine and cursor store")
    func scriptedConnectorFixtureDrivesRealEngine() async throws {
        let engine = try await IndexEngine.openInMemory()
        let cursorStore = FixtureCursorStore()
        let orchestrator = SyncOrchestrator(engine: engine, cursorStore: cursorStore)
        let events = FixturePayloads.basicNotes().map(SourceEvent.upsert) + [.checkpoint("fixture-cursor-1")]
        let connector = FixtureScriptedConnector(events: events)

        let outcome = try await orchestrator.sync(connector: connector, cursorKey: "fixture-key")
        let projection = SyncOutcomeProjection(outcome)

        #expect(projection.accepted == 3)
        #expect(projection.deletedCount == 0)
        #expect(projection.totalFailed == 0)
        #expect(projection.newCursor == "fixture-cursor-1")
        #expect(cursorStore.cursor(forKey: "fixture-key") == "fixture-cursor-1")

        let search = try await engine.search(.init(query: "compass navigation needle", mode: .diagnostic, limit: 3))
        #expect(SearchProjection(search).documentIDs.first == "fixture:compass")
    }
}
