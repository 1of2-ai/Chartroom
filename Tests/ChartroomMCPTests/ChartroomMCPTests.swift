import ChartroomControl
import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine
import Testing
@testable import ChartroomMCP

@Suite("Chartroom MCP")
struct ChartroomMCPTests {
    @Test("handles initialize, lists tools, and calls a real status tool")
    func handlesInitializeToolsListAndStatusCall() async throws {
        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        let server = ChartroomMCPServer(session: session)

        let initialize = try await server.handle(jsonLine: request(1, "initialize", [
            "protocolVersion": "2025-03-26",
            "clientInfo": ["name": "chartroom-test", "version": "1"]
        ]))
        let initializeObject = try object(from: initialize)
        #expect(initializeObject["id"] as? Int == 1)
        let initializeResult = try #require(initializeObject["result"] as? [String: Any])
        #expect(initializeResult["protocolVersion"] as? String == "2025-03-26")

        let toolsList = try await server.handle(jsonLine: request(2, "tools/list"))
        let toolsObject = try object(from: toolsList)
        let toolsResult = try #require(toolsObject["result"] as? [String: Any])
        let tools = try #require(toolsResult["tools"] as? [[String: Any]])
        let toolNames = Set(tools.compactMap { $0["name"] as? String })
        #expect(toolNames.isSuperset(of: [
            "chartroom_status",
            "chartroom_ingest_local_source",
            "chartroom_search",
            "chartroom_browse_documents",
            "chartroom_inspect_chunks",
            "chartroom_delete_documents"
        ]))

        _ = try await session.open()
        let status = try await server.handle(jsonLine: request(3, "tools/call", [
            "name": "chartroom_status",
            "arguments": [:]
        ]))
        let statusObject = try object(from: status)
        let statusResult = try #require(statusObject["result"] as? [String: Any])
        let content = try #require(statusResult["content"] as? [[String: Any]])
        let firstContent = try #require(content.first)
        let text = try #require(firstContent["text"] as? String)
        #expect(text.contains("\"state\" : \"ready\""))
    }

    @Test("calls local ingestion, search, browse, chunk inspection, and delete tools")
    func callsFeatureParityTools() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-mcp-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let noteURL = root.appending(path: "Remote.md")
        try "remote agent mcp feature parity needle".write(to: noteURL, atomically: true, encoding: .utf8)

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        let server = ChartroomMCPServer(session: session)
        _ = try await session.open()

        let ingestText = try await toolText(server, id: 10, name: "chartroom_ingest_local_source", arguments: [
            "path": root.path
        ])
        #expect(ingestText.contains("\"accepted\" : 1"))

        let searchText = try await toolText(server, id: 11, name: "chartroom_search", arguments: [
            "query": "feature parity needle",
            "mode": "diagnostic",
            "limit": 5
        ])
        #expect(searchText.contains("local-files:Remote.md"))

        let browseText = try await toolText(server, id: 12, name: "chartroom_browse_documents", arguments: [
            "limit": 10
        ])
        #expect(browseText.contains("Remote.md"))

        let chunksText = try await toolText(server, id: 13, name: "chartroom_inspect_chunks", arguments: [
            "documentID": "local-files:Remote.md"
        ])
        #expect(chunksText.contains("remote agent mcp feature parity"))

        let deleteText = try await toolText(server, id: 14, name: "chartroom_delete_documents", arguments: [
            "documentIDs": ["local-files:Remote.md"]
        ])
        #expect(deleteText.contains("\"deletedCount\" : 1"))
    }

    @Test("rejects local ingestion outside configured roots")
    func rejectsLocalIngestionOutsideAllowedRoots() async throws {
        let allowedRoot = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-mcp-allowed-\(UUID().uuidString)", directoryHint: .isDirectory)
        let deniedRoot = FileManager.default.temporaryDirectory
            .appending(path: "chartroom-mcp-denied-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: allowedRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: deniedRoot, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: allowedRoot)
            try? FileManager.default.removeItem(at: deniedRoot)
        }

        let session = ChartroomSession(
            engineFactory: { (try await IndexEngine.openInMemory(), nil) },
            cursorStore: InMemoryCursorStore()
        )
        let server = ChartroomMCPServer(session: session, allowedLocalRoots: [allowedRoot])
        _ = try await session.open()

        let line = try await server.handle(jsonLine: request(20, "tools/call", [
            "name": "chartroom_ingest_local_source",
            "arguments": ["path": deniedRoot.path]
        ]))
        let response = try object(from: line)
        let result = try #require(response["result"] as? [String: Any])
        #expect(result["isError"] as? Bool == true)
        let content = try #require(result["content"] as? [[String: Any]])
        let firstContent = try #require(content.first)
        let text = try #require(firstContent["text"] as? String)
        #expect(text.contains("outside the configured Chartroom roots"))
    }
}

private func request(_ id: Int, _ method: String, _ params: [String: Any]? = nil) throws -> String {
    var object: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
    if let params {
        object["params"] = params
    }
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

private func object(from line: String?) throws -> [String: Any] {
    let line = try #require(line)
    let data = try #require(line.data(using: .utf8))
    let decoded = try JSONSerialization.jsonObject(with: data)
    return try #require(decoded as? [String: Any])
}

private func toolText(
    _ server: ChartroomMCPServer,
    id: Int,
    name: String,
    arguments: [String: Any]
) async throws -> String {
    let line = try await server.handle(jsonLine: request(id, "tools/call", [
        "name": name,
        "arguments": arguments
    ]))
    let response = try object(from: line)
    let result = try #require(response["result"] as? [String: Any])
    let content = try #require(result["content"] as? [[String: Any]])
    let firstContent = try #require(content.first)
    return try #require(firstContent["text"] as? String)
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
