import ChartroomControl
import Foundation
import IndexEngine
import SyncEngine

public enum ChartroomMCPFactory {
    public static let storePathEnvironmentKey = "CHARTROOM_STORE_PATH"
    public static let allowedRootsEnvironmentKey = "CHARTROOM_ALLOWED_ROOTS"

    public static func defaultSession() throws -> ChartroomSession {
        let storeURL = try defaultStoreURL()
        return ChartroomSession(
            engineFactory: {
                let engine = try await IndexEngine.open(storeURL: storeURL)
                return (engine, storeURL)
            },
            cursorStore: UserDefaultsCursorStore(storageKey: "ChartroomMCP.connectorCursors.v1")
        )
    }

    private static func defaultStoreURL() throws -> URL {
        if let override = ProcessInfo.processInfo.environment[storePathEnvironmentKey], !override.isEmpty {
            let url = URL(fileURLWithPath: override)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            return url
        }

        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = base.appending(path: "Chartroom", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appending(path: "IndexEngine.sqlite")
    }

    public static func defaultAllowedLocalRoots() -> [URL] {
        if let override = ProcessInfo.processInfo.environment[allowedRootsEnvironmentKey], !override.isEmpty {
            return override
                .split(separator: ":")
                .map(String.init)
                .filter { !$0.isEmpty }
                .map { URL(fileURLWithPath: $0, isDirectory: true) }
        }

        return [FileManager.default.homeDirectoryForCurrentUser]
    }
}

public final class ChartroomMCPServer: @unchecked Sendable {
    private let session: ChartroomSession
    private let allowedLocalRoots: [URL]
    private let encoder: JSONEncoder

    public init(session: ChartroomSession, allowedLocalRoots: [URL] = []) {
        self.session = session
        self.allowedLocalRoots = allowedLocalRoots.map { $0.standardizedFileURL.resolvingSymlinksInPath() }
        self.encoder = JSONEncoder()
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        self.encoder.dateEncodingStrategy = .iso8601
    }

    public func handle(jsonLine: String) async throws -> String? {
        let data = Data(jsonLine.utf8)
        let decoded = try JSONSerialization.jsonObject(with: data)
        guard let request = decoded as? [String: Any] else {
            return try response(id: nil, error: -32600, message: "Invalid JSON-RPC request.")
        }

        let id = request["id"]
        guard let method = request["method"] as? String else {
            return id == nil ? nil : try response(id: id, error: -32600, message: "JSON-RPC request is missing method.")
        }

        if id == nil {
            return nil
        }

        switch method {
        case "initialize":
            let params = request["params"] as? [String: Any] ?? [:]
            return try response(id: id, result: [
                "protocolVersion": params["protocolVersion"] as? String ?? "2025-03-26",
                "capabilities": ["tools": [:]],
                "serverInfo": ["name": "chartroom-mcp", "version": "0.2.0"],
            ])
        case "tools/list":
            return try response(id: id, result: ["tools": tools])
        case "tools/call":
            let params = request["params"] as? [String: Any] ?? [:]
            guard let name = params["name"] as? String else {
                return try response(id: id, error: -32602, message: "tools/call requires a tool name.")
            }
            let arguments = params["arguments"] as? [String: Any] ?? [:]
            do {
                let text = try await callTool(name: name, arguments: arguments)
                return try response(id: id, result: toolResult(text: text, isError: false))
            } catch {
                return try response(id: id, result: toolResult(text: errorText(error), isError: true))
            }
        default:
            return try response(id: id, error: -32601, message: "Method not found: \(method)")
        }
    }

    private var tools: [[String: Any]] {
        [
            tool("chartroom_open", "Open the Chartroom index.", properties: [:]),
            tool("chartroom_status", "Read index, model, job, and failure status.", properties: [:]),
            tool("chartroom_search", "Search indexed content.", properties: [
                "query": stringSchema(description: "Search query."),
                "mode": enumSchema(values: ["fast", "quality", "diagnostic"], description: "Retrieval mode."),
                "limit": integerSchema(description: "Maximum results."),
                "sourceIDs": arraySchema(items: stringSchema(description: "Source ID."), description: "Optional source filters."),
                "contentTypes": arraySchema(items: stringSchema(description: "Content type."), description: "Optional content-type filters."),
            ], required: ["query"]),
            tool("chartroom_browse_documents", "Browse indexed documents.", properties: [
                "query": stringSchema(description: "Document filter query."),
                "limit": integerSchema(description: "Maximum documents."),
                "offset": integerSchema(description: "Pagination offset."),
                "sort": stringSchema(description: "DocumentSort raw value."),
            ]),
            tool("chartroom_inspect_chunks", "Inspect chunks for one document.", properties: [
                "documentID": stringSchema(description: "Document ID."),
            ], required: ["documentID"]),
            tool("chartroom_ingest_local_source", "Ingest a local file or folder.", properties: [
                "path": stringSchema(description: "Absolute local file or folder path."),
            ], required: ["path"]),
            tool("chartroom_delete_documents", "Delete indexed documents.", properties: [
                "documentIDs": arraySchema(items: stringSchema(description: "Document ID."), description: "Document IDs to delete."),
            ], required: ["documentIDs"]),
            tool("chartroom_list_failures", "List recorded failure diagnostics.", properties: [
                "limit": integerSchema(description: "Maximum failures."),
            ]),
            tool("chartroom_clear_failures", "Clear recorded failure diagnostics.", properties: [
                "ids": arraySchema(items: stringSchema(description: "Failure ID."), description: "Failure IDs. Omit to clear all failures."),
            ]),
            tool("chartroom_list_jobs", "List recorded jobs.", properties: [
                "limit": integerSchema(description: "Maximum jobs."),
            ]),
            tool("chartroom_benchmark", "Run a search benchmark.", properties: [
                "queries": arraySchema(items: stringSchema(description: "Benchmark query."), description: "Queries to benchmark."),
                "iterations": integerSchema(description: "Iterations per query."),
                "limit": integerSchema(description: "Search result limit."),
            ]),
            tool("chartroom_pipeline", "Read the active retrieval pipeline description.", properties: [:]),
            tool("chartroom_diagnostics_bundle", "Capture diagnostics as JSON.", properties: [
                "limit": integerSchema(description: "Maximum jobs and failures."),
            ]),
        ]
    }

    private func callTool(name: String, arguments: [String: Any]) async throws -> String {
        switch name {
        case "chartroom_open":
            return try encode(await session.open())
        case "chartroom_status":
            return try await encode(session.status(limit: int(arguments["limit"]) ?? 1_000))
        case "chartroom_search":
            let query = try requiredString(arguments["query"], name: "query")
            let mode = retrievalMode(arguments["mode"] as? String)
            let filters = SearchFilters(
                sourceIDs: Set(stringArray(arguments["sourceIDs"]).map(EngineID.init(rawValue:))),
                contentTypes: Set(stringArray(arguments["contentTypes"]))
            )
            return try await encode(session.search(.init(
                query: query,
                mode: mode,
                limit: int(arguments["limit"]) ?? 10,
                filters: filters
            )))
        case "chartroom_browse_documents":
            return try await encode(session.browseDocuments(.init(
                query: arguments["query"] as? String ?? "",
                sort: documentSort(arguments["sort"] as? String),
                limit: int(arguments["limit"]) ?? 50,
                offset: int(arguments["offset"]) ?? 0
            )))
        case "chartroom_inspect_chunks":
            let documentID = EngineID(rawValue: try requiredString(arguments["documentID"], name: "documentID"))
            return try await encode(session.chunks(forDocument: documentID))
        case "chartroom_ingest_local_source":
            let path = try requiredString(arguments["path"], name: "path")
            let outcome = try await session.ingestLocalSource(try validatedLocalURL(path))
            return try encode(SyncOutcomePayload(outcome: outcome))
        case "chartroom_delete_documents":
            let ids = stringArray(arguments["documentIDs"]).map(EngineID.init(rawValue:))
            guard !ids.isEmpty else {
                throw MCPToolError.invalidArguments("documentIDs must contain at least one ID.")
            }
            let summary = try await session.deleteDocuments(ids)
            return try encode(DeletionSummaryPayload(summary: summary))
        case "chartroom_list_failures":
            return try await encode(session.failures(limit: int(arguments["limit"]) ?? 1_000))
        case "chartroom_clear_failures":
            let ids = stringArray(arguments["ids"])
            try await session.clearFailures(ids: ids.isEmpty ? nil : Set(ids.map(EngineID.init(rawValue:))))
            return try encode(["cleared": true])
        case "chartroom_list_jobs":
            return try await encode(session.jobs(limit: int(arguments["limit"]) ?? 1_000))
        case "chartroom_benchmark":
            let queries = stringArray(arguments["queries"])
            let report = try await session.benchmark(
                queries: queries.isEmpty ? ["index", "search", "document"] : queries,
                iterations: int(arguments["iterations"]) ?? 5,
                limit: int(arguments["limit"]) ?? 10
            )
            return try encode(BenchmarkReportPayload(report: report))
        case "chartroom_pipeline":
            let pipeline = try await session.retrievalPipeline()
            return try encode(pipeline)
        case "chartroom_diagnostics_bundle":
            let bundle = try await session.diagnosticsBundle(limit: int(arguments["limit"]) ?? 1_000)
            return try encode(DiagnosticsPayload(bundle))
        default:
            throw MCPToolError.unknownTool(name)
        }
    }

    private func encode(_ value: some Encodable) throws -> String {
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    private func response(id: Any?, result: [String: Any]) throws -> String {
        try jsonLine(["jsonrpc": "2.0", "id": id ?? NSNull(), "result": result])
    }

    private func response(id: Any?, error code: Int, message: String) throws -> String {
        try jsonLine([
            "jsonrpc": "2.0",
            "id": id ?? NSNull(),
            "error": ["code": code, "message": message],
        ])
    }

    private func toolResult(text: String, isError: Bool) -> [String: Any] {
        [
            "content": [["type": "text", "text": text]],
            "isError": isError,
        ]
    }

    private func jsonLine(_ object: [String: Any]) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private func errorText(_ error: Error) -> String {
        if let engineError = error as? IndexEngineError {
            return "\(engineError.code): \(engineError.summary)"
        }
        return String(describing: error)
    }

    private func validatedLocalURL(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard !allowedLocalRoots.isEmpty else { return url }

        let isAllowed = allowedLocalRoots.contains { root in
            url.path == root.path || url.path.hasPrefix(root.path + "/")
        }
        guard isAllowed else {
            throw MCPToolError.invalidArguments("Local source is outside the configured Chartroom roots.")
        }
        return url
    }
}

private struct SyncOutcomePayload: Codable {
    var payloadCount: Int
    var accepted: Int
    var ingestFailed: Int
    var deletedCount: Int
    var deleteFailed: Int
    var stopped: Bool
    var newCursor: String?
    var hadChanges: Bool
    var unreadablePaths: [String]
    var unreadableKeptCount: Int
    var totalFailed: Int

    init(outcome: SyncOutcome) {
        payloadCount = outcome.payloadCount
        accepted = outcome.accepted
        ingestFailed = outcome.ingestFailed
        deletedCount = outcome.deletedCount
        deleteFailed = outcome.deleteFailed
        stopped = outcome.stopped
        newCursor = outcome.newCursor
        hadChanges = outcome.hadChanges
        unreadablePaths = outcome.unreadablePaths
        unreadableKeptCount = outcome.unreadableKeptCount
        totalFailed = outcome.totalFailed
    }
}

private struct DeletionSummaryPayload: Codable {
    var jobID: String
    var requestedCount: Int
    var deletedCount: Int
    var failedCount: Int
    var failures: [FailureSnapshot]
    var startedAt: Date
    var finishedAt: Date

    init(summary: DeletionSummary) {
        jobID = summary.jobID.rawValue
        requestedCount = summary.requestedCount
        deletedCount = summary.deletedCount
        failedCount = summary.failedCount
        failures = summary.failures
        startedAt = summary.startedAt
        finishedAt = summary.finishedAt
    }
}

private struct BenchmarkReportPayload: Codable {
    struct Layer: Codable {
        var id: String
        var p50: TimeInterval?
        var p95: TimeInterval?
        var sampleCount: Int
    }

    var queries: [String]
    var iterations: Int
    var totalRuns: Int
    var layers: [Layer]

    init(report: SearchBenchmark.Report) {
        queries = report.queries
        iterations = report.iterations
        totalRuns = report.totalRuns
        layers = report.layers.map {
            Layer(id: $0.id, p50: $0.p50, p95: $0.p95, sampleCount: $0.sampleCount)
        }
    }
}

private struct DiagnosticsPayload: Codable {
    var snapshot: IndexEngineSnapshot
    var health: IndexHealthSnapshot
    var modelStatus: ModelStatusSnapshot
    var jobs: [JobSnapshot]
    var failures: [FailureSnapshot]
    var lastSearch: SearchResponse?

    init(_ bundle: DiagnosticsBundle) {
        snapshot = bundle.snapshot
        health = bundle.health
        modelStatus = bundle.modelStatus
        jobs = bundle.jobs
        failures = bundle.failures
        lastSearch = bundle.lastSearch
    }
}

private enum MCPToolError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case unknownTool(String)

    var description: String {
        switch self {
        case let .invalidArguments(message):
            message
        case let .unknownTool(name):
            "Unknown tool: \(name)"
        }
    }
}

private func tool(
    _ name: String,
    _ description: String,
    properties: [String: Any],
    required: [String] = []
) -> [String: Any] {
    [
        "name": name,
        "description": description,
        "inputSchema": [
            "type": "object",
            "properties": properties,
            "required": required,
        ],
    ]
}

private func stringSchema(description: String) -> [String: Any] {
    ["type": "string", "description": description]
}

private func integerSchema(description: String) -> [String: Any] {
    ["type": "integer", "description": description]
}

private func enumSchema(values: [String], description: String) -> [String: Any] {
    ["type": "string", "enum": values, "description": description]
}

private func arraySchema(items: [String: Any], description: String) -> [String: Any] {
    ["type": "array", "items": items, "description": description]
}

private func requiredString(_ value: Any?, name: String) throws -> String {
    guard let string = value as? String, !string.isEmpty else {
        throw MCPToolError.invalidArguments("\(name) must be a non-empty string.")
    }
    return string
}

private func stringArray(_ value: Any?) -> [String] {
    (value as? [String]) ?? []
}

private func int(_ value: Any?) -> Int? {
    switch value {
    case let int as Int:
        int
    case let double as Double:
        Int(double)
    case let string as String:
        Int(string)
    default:
        nil
    }
}

private func retrievalMode(_ value: String?) -> RetrievalMode {
    guard let value, let mode = RetrievalMode(rawValue: value) else {
        return .fast
    }
    return mode
}

private func documentSort(_ value: String?) -> DocumentSort {
    guard let value, let sort = DocumentSort(rawValue: value) else {
        return .ingestedAtDescending
    }
    return sort
}
