import Foundation

public struct IndexedObject: Sendable, Equatable {
    public var id: String
    public var type: String
    public var title: String
    public var body: String
    public var clusterID: String?
    public var sourceID: String?
    public var sourceURI: URL?
    public var policyID: String?
    public var representationID: String?
    public var embeddingSpaceID: String?

    public init(
        id: String,
        type: String,
        title: String,
        body: String,
        clusterID: String? = nil,
        sourceID: String? = nil,
        sourceURI: URL? = nil,
        policyID: String? = nil,
        representationID: String? = nil,
        embeddingSpaceID: String? = nil
    ) {
        self.id = id
        self.type = type
        self.title = title
        self.body = body
        self.clusterID = clusterID
        self.sourceID = sourceID
        self.sourceURI = sourceURI
        self.policyID = policyID
        self.representationID = representationID
        self.embeddingSpaceID = embeddingSpaceID
    }
}

public struct Scope: Sendable {
    public var clusterID: String?
    public var boostInScope: Double
    public var hardScope: Bool

    public init(clusterID: String? = nil, boostInScope: Double = 2.0, hardScope: Bool = false) {
        self.clusterID = clusterID
        self.boostInScope = boostInScope
        self.hardScope = hardScope
    }

    public static let global = Scope()

    public static func cluster(_ id: String, hard: Bool = false) -> Scope {
        Scope(clusterID: id, boostInScope: 2.0, hardScope: hard)
    }
}

public struct SearchHit: Sendable, Equatable {
    public var id: String
    public var documentID: String
    public var chunkID: String
    public var type: String
    public var title: String
    public var snippet: String
    public var sourceID: String?
    public var sourceURI: URL?
    public var policyID: String?
    public var representationID: String?
    public var embeddingSpaceID: String?
    public var score: Double
    public var exactRank: Int?
    public var keywordRank: Int?
    public var vectorRank: Int?
}

public struct IndexStoreCounts: Sendable, Equatable {
    public var documentCount: Int
    public var chunkCount: Int
    public var embeddingCount: Int

    public init(documentCount: Int, chunkCount: Int, embeddingCount: Int) {
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.embeddingCount = embeddingCount
    }
}

public enum IndexStoreError: Error, CustomStringConvertible, Equatable {
    case embeddingDimensionMismatch(kind: EmbedKind, expected: Int, actual: Int)
    case storedVectorDimensionMismatch(id: String, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .embeddingDimensionMismatch(kind, expected, actual):
            "Embedding dimension mismatch for \(kind): expected \(expected), got \(actual)"
        case let .storedVectorDimensionMismatch(id, expected, actual):
            "Stored vector dimension mismatch for \(id): expected \(expected), got \(actual)"
        }
    }
}

struct CandidateFilter: Sendable {
    var whereSQL: String
    var bindings: [String]
}

public actor IndexStore {
    let db: SQLite
    let embedder: any Embedder
    public let modelID: String
    public let dimension: Int
    public let embeddingSpaceID: String
    public let vectorBackendID = EngineID.builtInSQLiteVectorBackend.rawValue
    public let vectorBackendVersion = "1"

    var upsertGenerations: [String: UInt64] = [:]
    /// Confirmed-available probe result; see `embeddingProviderStatus()`.
    var cachedEmbeddingProviderStatus: (isAvailable: Bool, message: String)?
    /// The `k` constant in the reciprocal-rank-fusion weight `1 / (k + rank)`. Public so the
    /// engine can report its real fusion parameter through `RetrievalPipelineDescriptor`
    /// rather than the GUI hardcoding a copy that could drift.
    public static let reciprocalRankK = 60.0
    let defaultChunkCharacterLimit = 1_600
    let defaultChunkOverlap = 160

    public init(path: String, embedder: any Embedder) throws {
        self.db = try SQLite(path: path)
        self.embedder = embedder
        self.modelID = embedder.modelID
        self.dimension = embedder.dimension
        self.embeddingSpaceID = embedder.embeddingSpaceID
        try Self.installSchema(
            db: db,
            vectorBackendID: vectorBackendID,
            vectorBackendVersion: vectorBackendVersion
        )
    }

    /// On-disk footprint of the store — the SQLite database plus its WAL/SHM sidecars —
    /// or nil for an in-memory store. The storage format is the store's concern; callers
    /// (snapshots, diagnostics) report the number without knowing the file layout.
    public func storeByteSize() -> Int64? {
        db.fileByteSize
    }

    func validateEmbedding(_ vector: [Float], kind: EmbedKind) throws {
        guard vector.count == dimension else {
            throw IndexStoreError.embeddingDimensionMismatch(kind: kind, expected: dimension, actual: vector.count)
        }
    }

    static func placeholders(count: Int) -> String {
        Array(repeating: "?", count: count).joined(separator: ",")
    }

    static func likePattern(for query: String) -> String {
        let escaped = query
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "%", with: "\\%")
            .replacingOccurrences(of: "_", with: "\\_")
        return "%\(escaped)%"
    }
}

/// Multi-field FNV-1a with a separator round per field, stable across processes
/// (Swift's `Hasher` is per-process seeded). Public because connectors fingerprint
/// files with the same hasher; outputs are persisted, so the algorithm must not drift.
public struct StableFNV1A {
    public private(set) var value: UInt64 = 0xcbf29ce484222325

    public init() {}

    public mutating func update(_ string: String) {
        for byte in string.utf8 {
            value = (value ^ UInt64(byte)) &* 0x100000001b3
        }
        value = (value ^ 0xff) &* 0x100000001b3
    }
}

func emptyStringAsNil(_ value: String?) -> String? {
    guard let value, !value.isEmpty else { return nil }
    return value
}
