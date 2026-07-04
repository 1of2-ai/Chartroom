import Foundation

public struct DocumentBrowseResponse: Codable, Hashable, Sendable {
    public var request: DocumentBrowseRequest
    public var documents: [DocumentSummary]
    public var totalMatching: Int
    public var facets: DocumentBrowseFacets

    public init(
        request: DocumentBrowseRequest,
        documents: [DocumentSummary],
        totalMatching: Int,
        facets: DocumentBrowseFacets = .empty
    ) {
        self.request = request
        self.documents = documents
        self.totalMatching = totalMatching
        self.facets = facets
    }

    public var returnedCount: Int {
        documents.count
    }

    public var offset: Int {
        request.offset
    }

    public var startIndex: Int? {
        documents.isEmpty ? nil : offset + 1
    }

    public var endIndex: Int? {
        documents.isEmpty ? nil : offset + returnedCount
    }

    public var hasPreviousPage: Bool {
        offset > 0
    }

    public var hasNextPage: Bool {
        offset + returnedCount < totalMatching
    }

    public var isTruncated: Bool {
        hasNextPage
    }
}

public struct DocumentBrowseFacets: Codable, Hashable, Sendable {
    public static let empty = DocumentBrowseFacets(sourceIDs: [], contentTypes: [])

    public var sourceIDs: [SourceID]
    public var contentTypes: [String]

    public init(sourceIDs: [SourceID], contentTypes: [String]) {
        self.sourceIDs = sourceIDs
        self.contentTypes = contentTypes
    }
}

/// A typed, GUI-facing projection of one stored document. It is enough to list, filter, and inspect
/// the corpus without exposing storage internals. Counts and identity come from durable records; the
/// body stays behind `search` and future chunk inspection contracts.
public struct DocumentSummary: Codable, Hashable, Sendable, Identifiable {
    public var id: DocumentID
    public var title: String
    public var sourceID: SourceID?
    public var sourceURI: URL?
    public var contentType: String
    public var byteSize: Int
    public var chunkCount: Int
    public var ingestedAt: Date
    public var modifiedAt: Date?
    public var policyID: PolicyID?
    public var clusterID: EngineID?

    public init(
        id: DocumentID,
        title: String,
        sourceID: SourceID? = nil,
        sourceURI: URL? = nil,
        contentType: String,
        byteSize: Int,
        chunkCount: Int,
        ingestedAt: Date,
        modifiedAt: Date? = nil,
        policyID: PolicyID? = nil,
        clusterID: EngineID? = nil
    ) {
        self.id = id
        self.title = title
        self.sourceID = sourceID
        self.sourceURI = sourceURI
        self.contentType = contentType
        self.byteSize = byteSize
        self.chunkCount = chunkCount
        self.ingestedAt = ingestedAt
        self.modifiedAt = modifiedAt
        self.policyID = policyID
        self.clusterID = clusterID
    }
}

/// A typed, GUI-facing projection of one active chunk — enough for a retrieval harness
/// to browse a document's chunks (text, ordinal, heading path, offset ranges) and see
/// whether each one carries an embedding.
public struct ChunkSummary: Codable, Hashable, Sendable, Identifiable {
    public var id: ChunkID
    public var documentID: DocumentID
    public var ordinal: Int
    public var text: String
    public var headingPath: String
    public var byteStart: Int
    public var byteEnd: Int
    public var characterStart: Int
    public var characterEnd: Int
    public var tokenStart: Int
    public var tokenEnd: Int
    public var contentHash: String
    public var hasEmbedding: Bool

    public init(
        id: ChunkID,
        documentID: DocumentID,
        ordinal: Int,
        text: String,
        headingPath: String = "",
        byteStart: Int,
        byteEnd: Int,
        characterStart: Int,
        characterEnd: Int,
        tokenStart: Int,
        tokenEnd: Int,
        contentHash: String,
        hasEmbedding: Bool
    ) {
        self.id = id
        self.documentID = documentID
        self.ordinal = ordinal
        self.text = text
        self.headingPath = headingPath
        self.byteStart = byteStart
        self.byteEnd = byteEnd
        self.characterStart = characterStart
        self.characterEnd = characterEnd
        self.tokenStart = tokenStart
        self.tokenEnd = tokenEnd
        self.contentHash = contentHash
        self.hasEmbedding = hasEmbedding
    }
}

public struct IndexEngineSnapshot: Codable, Sendable, Equatable {
    public var storeURL: URL?
    /// On-disk footprint of the store in bytes (database + WAL/SHM sidecars), or nil for
    /// an in-memory store or when the size cannot be read. Optional so older diagnostics
    /// JSON without the field still decodes.
    public var storeByteSize: Int64?
    public var objectCount: Int
    public var documentCount: Int
    public var chunkCount: Int
    public var embeddingCount: Int
    public var modelID: String
    public var embeddingDimension: Int
    public var embeddingSpaceID: EmbeddingSpaceID
    public var lastIngestedAt: Date?
    public var policyStates: [PolicyResolution]

    public init(
        storeURL: URL?,
        storeByteSize: Int64? = nil,
        objectCount: Int,
        documentCount: Int = 0,
        chunkCount: Int = 0,
        embeddingCount: Int = 0,
        modelID: String,
        embeddingDimension: Int,
        embeddingSpaceID: EmbeddingSpaceID,
        lastIngestedAt: Date?,
        policyStates: [PolicyResolution]
    ) {
        self.storeURL = storeURL
        self.storeByteSize = storeByteSize
        self.objectCount = objectCount
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.embeddingCount = embeddingCount
        self.modelID = modelID
        self.embeddingDimension = embeddingDimension
        self.embeddingSpaceID = embeddingSpaceID
        self.lastIngestedAt = lastIngestedAt
        self.policyStates = policyStates
    }
}

public struct IngestionSummary: Sendable, Equatable {
    public var jobID: JobID
    public var acceptedCount: Int
    public var failedCount: Int
    public var failures: [FailureSnapshot]
    public var startedAt: Date
    public var finishedAt: Date

    public init(
        jobID: JobID,
        acceptedCount: Int,
        failedCount: Int,
        failures: [FailureSnapshot],
        startedAt: Date,
        finishedAt: Date
    ) {
        self.jobID = jobID
        self.acceptedCount = acceptedCount
        self.failedCount = failedCount
        self.failures = failures
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }
}

public struct DeletionSummary: Sendable, Equatable {
    public var jobID: JobID
    public var requestedCount: Int
    public var deletedCount: Int
    public var failedCount: Int
    public var failures: [FailureSnapshot]
    public var startedAt: Date
    public var finishedAt: Date

    public init(
        jobID: JobID,
        requestedCount: Int,
        deletedCount: Int,
        failedCount: Int,
        failures: [FailureSnapshot],
        startedAt: Date,
        finishedAt: Date
    ) {
        self.jobID = jobID
        self.requestedCount = requestedCount
        self.deletedCount = deletedCount
        self.failedCount = failedCount
        self.failures = failures
        self.startedAt = startedAt
        self.finishedAt = finishedAt
    }
}

public struct SearchResponse: Codable, Sendable, Equatable {
    public var query: String
    public var mode: RetrievalMode
    public var results: [SearchResultSnapshot]
    public var diagnostics: SearchDiagnostics

    public init(
        query: String,
        mode: RetrievalMode,
        results: [SearchResultSnapshot],
        diagnostics: SearchDiagnostics
    ) {
        self.query = query
        self.mode = mode
        self.results = results
        self.diagnostics = diagnostics
    }
}

public struct SearchResultSnapshot: Codable, Hashable, Sendable {
    public var id: EngineID
    public var documentID: DocumentID
    public var chunkID: ChunkID
    public var sourceID: SourceID?
    public var title: String
    public var snippet: String?
    public var sourceURI: URL?
    public var contentType: String
    public var score: Double
    public var rank: Int
    public var diagnostics: SearchResultDiagnostics
    public var provenance: ResultProvenance

    public init(
        id: EngineID,
        documentID: DocumentID,
        chunkID: ChunkID,
        sourceID: SourceID?,
        title: String,
        snippet: String?,
        sourceURI: URL?,
        contentType: String,
        score: Double,
        rank: Int,
        diagnostics: SearchResultDiagnostics,
        provenance: ResultProvenance
    ) {
        self.id = id
        self.documentID = documentID
        self.chunkID = chunkID
        self.sourceID = sourceID
        self.title = title
        self.snippet = snippet
        self.sourceURI = sourceURI
        self.contentType = contentType
        self.score = score
        self.rank = rank
        self.diagnostics = diagnostics
        self.provenance = provenance
    }
}

public struct SearchDiagnostics: Codable, Hashable, Sendable {
    public var degraded: Bool
    public var missingChannels: [RetrievalChannel]
    public var sqlFilterLatency: TimeInterval?
    public var ftsLatency: TimeInterval?
    public var vectorLatency: TimeInterval?
    public var fusionLatency: TimeInterval?
    public var snippetLatency: TimeInterval?
    public var totalLatency: TimeInterval?

    public init(
        degraded: Bool = false,
        missingChannels: [RetrievalChannel] = [],
        sqlFilterLatency: TimeInterval? = nil,
        ftsLatency: TimeInterval? = nil,
        vectorLatency: TimeInterval? = nil,
        fusionLatency: TimeInterval? = nil,
        snippetLatency: TimeInterval? = nil,
        totalLatency: TimeInterval? = nil
    ) {
        self.degraded = degraded
        self.missingChannels = missingChannels
        self.sqlFilterLatency = sqlFilterLatency
        self.ftsLatency = ftsLatency
        self.vectorLatency = vectorLatency
        self.fusionLatency = fusionLatency
        self.snippetLatency = snippetLatency
        self.totalLatency = totalLatency
    }
}

public enum RetrievalChannel: String, Codable, Hashable, Sendable {
    case sql
    case fts
    case vector
    case exact
    case graph
    case reranker
}

public struct SearchResultDiagnostics: Codable, Hashable, Sendable {
    public var ftsRank: Int?
    public var vectorRank: Int?
    public var exactRank: Int?
    public var graphReason: String?
    public var appliedBoosts: [AppliedBoost]

    public init(
        ftsRank: Int? = nil,
        vectorRank: Int? = nil,
        exactRank: Int? = nil,
        graphReason: String? = nil,
        appliedBoosts: [AppliedBoost] = []
    ) {
        self.ftsRank = ftsRank
        self.vectorRank = vectorRank
        self.exactRank = exactRank
        self.graphReason = graphReason
        self.appliedBoosts = appliedBoosts
    }
}

public struct AppliedBoost: Codable, Hashable, Sendable {
    public var id: EngineID
    public var label: String
    public var value: Double

    public init(id: EngineID, label: String, value: Double) {
        self.id = id
        self.label = label
        self.value = value
    }
}

public struct ResultProvenance: Codable, Hashable, Sendable {
    public var connectorID: ConnectorID?
    public var policyID: PolicyID?
    public var representationID: RepresentationID?
    public var embeddingSpaceID: EmbeddingSpaceID?

    public init(
        connectorID: ConnectorID? = nil,
        policyID: PolicyID? = nil,
        representationID: RepresentationID? = nil,
        embeddingSpaceID: EmbeddingSpaceID? = nil
    ) {
        self.connectorID = connectorID
        self.policyID = policyID
        self.representationID = representationID
        self.embeddingSpaceID = embeddingSpaceID
    }
}

public struct FailureSnapshot: Codable, Hashable, Sendable {
    public enum Category: String, Codable, Sendable {
        case sourceUnavailable
        case permissionDenied
        case unsupportedContentType
        case decodeFailure
        case extractionFailure
        case chunkingFailure
        case embeddingFailure
        case storageFailure
        case migrationFailure
        case connectorProtocolFailure
        case mcpFailure
    }

    public var id: EngineID
    public var category: Category
    public var message: String
    public var detail: String
    public var sourceID: SourceID?
    public var documentID: DocumentID?
    /// Where the failed payload came from, so retries can re-fetch it directly
    /// instead of reconstructing a location from the document ID.
    public var sourceURI: URL?
    public var recoverability: IndexEngineError.Recoverability
    public var occurredAt: Date
    public var isRecoverable: Bool { recoverability != .unrecoverable }

    public init(
        id: EngineID,
        category: Category,
        message: String,
        detail: String,
        sourceID: SourceID? = nil,
        documentID: DocumentID? = nil,
        sourceURI: URL? = nil,
        recoverability: IndexEngineError.Recoverability,
        occurredAt: Date
    ) {
        self.id = id
        self.category = category
        self.message = message
        self.detail = detail
        self.sourceID = sourceID
        self.documentID = documentID
        self.sourceURI = sourceURI
        self.recoverability = recoverability
        self.occurredAt = occurredAt
    }

    public init(
        id: EngineID,
        category: Category,
        message: String,
        detail: String,
        sourceID: SourceID? = nil,
        documentID: DocumentID? = nil,
        isRecoverable: Bool,
        occurredAt: Date
    ) {
        self.init(
            id: id,
            category: category,
            message: message,
            detail: detail,
            sourceID: sourceID,
            documentID: documentID,
            recoverability: isRecoverable ? .retryable : .unrecoverable,
            occurredAt: occurredAt
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case category
        case message
        case detail
        case sourceID
        case documentID
        case sourceURI
        case recoverability
        case isRecoverable
        case occurredAt
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(EngineID.self, forKey: .id)
        category = try container.decode(Category.self, forKey: .category)
        message = try container.decode(String.self, forKey: .message)
        detail = try container.decode(String.self, forKey: .detail)
        sourceID = try container.decodeIfPresent(SourceID.self, forKey: .sourceID)
        documentID = try container.decodeIfPresent(DocumentID.self, forKey: .documentID)
        sourceURI = try container.decodeIfPresent(URL.self, forKey: .sourceURI)
        occurredAt = try container.decode(Date.self, forKey: .occurredAt)

        if let storedRecoverability = try container.decodeIfPresent(
            IndexEngineError.Recoverability.self,
            forKey: .recoverability
        ) {
            recoverability = storedRecoverability
        } else {
            let legacyIsRecoverable = try container.decodeIfPresent(Bool.self, forKey: .isRecoverable) ?? true
            recoverability = legacyIsRecoverable ? .retryable : .unrecoverable
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(category, forKey: .category)
        try container.encode(message, forKey: .message)
        try container.encode(detail, forKey: .detail)
        try container.encodeIfPresent(sourceID, forKey: .sourceID)
        try container.encodeIfPresent(documentID, forKey: .documentID)
        try container.encodeIfPresent(sourceURI, forKey: .sourceURI)
        try container.encode(recoverability, forKey: .recoverability)
        try container.encode(isRecoverable, forKey: .isRecoverable)
        try container.encode(occurredAt, forKey: .occurredAt)
    }
}

public struct IndexHealthSnapshot: Codable, Hashable, Sendable {
    public var objectCount: Int
    public var documentCount: Int
    public var chunkCount: Int
    public var embeddingCount: Int
    public var policyStates: [PolicyResolution]
    public var vectorBackendStatus: VectorStorageStatus?

    public init(
        objectCount: Int,
        documentCount: Int = 0,
        chunkCount: Int = 0,
        embeddingCount: Int = 0,
        policyStates: [PolicyResolution],
        vectorBackendStatus: VectorStorageStatus? = nil
    ) {
        self.objectCount = objectCount
        self.documentCount = documentCount
        self.chunkCount = chunkCount
        self.embeddingCount = embeddingCount
        self.policyStates = policyStates
        self.vectorBackendStatus = vectorBackendStatus
    }
}

public struct JobSnapshot: Codable, Hashable, Sendable {
    public enum State: String, Codable, Sendable {
        case queued
        case running
        case committing
        case succeeded
        case failed
        case cancelled
        case recovering
    }

    public enum Kind: String, Codable, Sendable {
        case ingest
        case delete
    }

    public var id: JobID
    public var state: State
    public var kind: Kind
    public var completedUnitCount: Int
    public var totalUnitCount: Int?
    public var message: String

    public init(
        id: JobID,
        state: State,
        kind: Kind = .ingest,
        completedUnitCount: Int = 0,
        totalUnitCount: Int? = nil,
        message: String = ""
    ) {
        self.id = id
        self.state = state
        self.kind = kind
        self.completedUnitCount = completedUnitCount
        self.totalUnitCount = totalUnitCount
        self.message = message
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case state
        case kind
        case completedUnitCount
        case totalUnitCount
        case message
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(JobID.self, forKey: .id)
        state = try container.decode(State.self, forKey: .state)
        kind = try container.decodeIfPresent(Kind.self, forKey: .kind) ?? .ingest
        completedUnitCount = try container.decode(Int.self, forKey: .completedUnitCount)
        totalUnitCount = try container.decodeIfPresent(Int.self, forKey: .totalUnitCount)
        message = try container.decode(String.self, forKey: .message)
    }
}

public struct ModelStatusSnapshot: Codable, Hashable, Sendable {
    public var modelID: String
    public var embeddingSpaceID: EmbeddingSpaceID?
    public var dimension: Int
    public var isAvailable: Bool
    public var message: String

    public init(
        modelID: String,
        embeddingSpaceID: EmbeddingSpaceID?,
        dimension: Int,
        isAvailable: Bool,
        message: String = ""
    ) {
        self.modelID = modelID
        self.embeddingSpaceID = embeddingSpaceID
        self.dimension = dimension
        self.isAvailable = isAvailable
        self.message = message
    }
}

public struct VectorStorageStatus: Codable, Hashable, Sendable {
    public enum State: String, Codable, Sendable {
        case unavailable
        case preparing
        case ready
        case degraded
        case failed
    }

    public var backendID: VectorBackendID
    public var state: State
    public var message: String

    public init(backendID: VectorBackendID, state: State, message: String = "") {
        self.backendID = backendID
        self.state = state
        self.message = message
    }
}
