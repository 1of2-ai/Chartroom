import ConnectorEngine
import Foundation
import IndexEngine
import SyncEngine

public struct FixtureFile: Sendable {
    public var relativePath: String
    public var contents: String

    public init(relativePath: String, contents: String) {
        self.relativePath = relativePath
        self.contents = contents
    }
}

public final class FixtureWorkspace: @unchecked Sendable {
    public let rootURL: URL

    public init(name: String) throws {
        rootURL = FileManager.default.temporaryDirectory
            .appending(path: "\(name)-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
    }

    public func write(_ file: FixtureFile) throws -> URL {
        try write(file.relativePath, contents: file.contents)
    }

    @discardableResult
    public func write(_ relativePath: String, contents: String) throws -> URL {
        let fileURL = url(relativePath)
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try contents.write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    public func remove(_ relativePath: String) throws {
        let fileURL = url(relativePath)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            try FileManager.default.removeItem(at: fileURL)
        }
    }

    public func url(_ relativePath: String) -> URL {
        rootURL.appending(path: relativePath)
    }

    public func cleanup() {
        try? FileManager.default.removeItem(at: rootURL)
    }
}

public enum FixtureCorpora {
    public static let basicNotes: [FixtureFile] = [
        FixtureFile(
            relativePath: "Atlas.md",
            contents: """
            # Atlas

            Atlas routing needle explains retrieval paths, source filters, and stable search projections.
            """
        ),
        FixtureFile(
            relativePath: "Beacon.md",
            contents: """
            # Beacon

            Beacon observability needle covers diagnostic jobs, failures, and health snapshots.
            """
        ),
        FixtureFile(
            relativePath: "Research/Compass.md",
            contents: """
            # Compass

            Compass navigation needle covers nested paths, browse sorting, and chunk inspection.
            """
        ),
        FixtureFile(
            relativePath: "Unicode.md",
            contents: """
            # Unicode

            Café résumé naïve unicode needle keeps snippet handling honest.
            """
        ),
    ]

    public static let mutationSeed: [FixtureFile] = [
        FixtureFile(relativePath: "Stable.md", contents: "Stable baseline mutation needle stays indexed."),
        FixtureFile(relativePath: "Removed.md", contents: "Removed baseline mutation needle should disappear."),
    ]

    public static func writeBasicNotes() throws -> FixtureWorkspace {
        let workspace = try FixtureWorkspace(name: "chartroom-basic-notes")
        for file in basicNotes {
            _ = try workspace.write(file)
        }
        return workspace
    }

    public static func writeMutationSeed() throws -> FixtureWorkspace {
        let workspace = try FixtureWorkspace(name: "chartroom-mutations")
        for file in mutationSeed {
            _ = try workspace.write(file)
        }
        return workspace
    }
}

public enum FixturePayloads {
    public static func basicNotes(sourceID: SourceID = "fixture-source") -> [SourcePayload] {
        [
            payload(
                id: "fixture:atlas",
                sourceID: sourceID,
                displayName: "Atlas",
                body: "Atlas routing needle explains retrieval paths and source filters.",
                clusterID: "fixture-cluster-routing"
            ),
            payload(
                id: "fixture:beacon",
                sourceID: sourceID,
                displayName: "Beacon",
                body: "Beacon observability needle covers diagnostic jobs and failures.",
                clusterID: "fixture-cluster-diagnostics"
            ),
            payload(
                id: "fixture:compass",
                sourceID: sourceID,
                displayName: "Compass",
                body: "Compass navigation needle covers nested browse and chunk inspection.",
                clusterID: "fixture-cluster-routing"
            ),
        ]
    }

    public static func payload(
        id: DocumentID,
        sourceID: SourceID = "fixture-source",
        displayName: String? = nil,
        body: String,
        contentType: String = "public.plain-text",
        clusterID: EngineID? = nil
    ) -> SourcePayload {
        SourcePayload(
            documentID: id,
            sourceID: sourceID,
            sourceURI: URL(filePath: "/fixtures/\(id.rawValue)"),
            displayName: displayName ?? id.rawValue,
            contentType: contentType,
            body: .text(body),
            clusterID: clusterID
        )
    }
}

public final class FixtureCursorStore: CursorStore, @unchecked Sendable {
    private let lock = NSLock()
    private var cursors: [String: SourceCursor] = [:]

    public init() {}

    public func cursor(forKey key: String) -> SourceCursor? {
        lock.withLock { cursors[key] }
    }

    public func setCursor(_ cursor: SourceCursor, forKey key: String) {
        lock.withLock {
            cursors[key] = cursor
        }
    }
}

public final class FixtureScriptedConnector: SourceConnector, @unchecked Sendable {
    public let id: ConnectorID
    public let capabilities: ConnectorCapabilities

    private let events: [SourceEvent]
    private let lock = NSLock()
    private var lastCursor: SourceCursor?

    public init(
        id: ConnectorID = "fixture-scripted",
        events: [SourceEvent],
        capabilities: ConnectorCapabilities = .init()
    ) {
        self.id = id
        self.events = events
        self.capabilities = capabilities
    }

    public func validate() async throws {}

    public func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error> {
        lock.withLock {
            lastCursor = cursor
        }
        let events = events
        return AsyncThrowingStream { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }

    public func fetch(_ reference: SourceReference) async throws -> SourcePayload {
        throw IndexEngineError(
            .sourceRefetchRequired,
            code: "fixture.fetch.unsupported",
            recoverability: .unrecoverable,
            summary: "FixtureScriptedConnector does not support fetch.",
            detail: reference.uri.absoluteString
        )
    }

    public func receivedCursor() -> SourceCursor? {
        lock.withLock { lastCursor }
    }
}

public struct SearchProjection: Equatable, Sendable {
    public var query: String
    public var documentIDs: [String]
    public var snippets: [String]
    public var sourceIDs: [String]

    public init(_ response: SearchResponse) {
        query = response.query
        documentIDs = response.results.map(\.documentID.rawValue)
        snippets = response.results.compactMap(\.snippet)
        sourceIDs = response.results.compactMap { $0.sourceID?.rawValue }
    }
}

public struct BrowseProjection: Equatable, Sendable {
    public var documentIDs: [String]
    public var titles: [String]
    public var totalMatching: Int

    public init(_ response: DocumentBrowseResponse) {
        documentIDs = response.documents.map(\.id.rawValue)
        titles = response.documents.map(\.title)
        totalMatching = response.totalMatching
    }
}

public struct ChunkProjection: Equatable, Sendable {
    public var documentIDs: [String]
    public var texts: [String]

    public init(_ chunks: [ChunkSummary]) {
        documentIDs = chunks.map(\.documentID.rawValue)
        texts = chunks.map(\.text)
    }
}

public struct SyncOutcomeProjection: Equatable, Sendable {
    public var accepted: Int
    public var deletedCount: Int
    public var totalFailed: Int
    public var newCursor: SourceCursor?

    public init(_ outcome: SyncOutcome) {
        accepted = outcome.accepted
        deletedCount = outcome.deletedCount
        totalFailed = outcome.totalFailed
        newCursor = outcome.newCursor
    }
}
