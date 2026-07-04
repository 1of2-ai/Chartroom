import Foundation
import IndexEngine

/// Opaque, connector-owned checkpoint token.
///
/// `IndexEngine` does not interpret cursors. A connector advances this value
/// only after its emitted payloads have been durably accepted by the engine.
public typealias SourceCursor = String

/// Normalized source change stream for local files, directory walks, MCP
/// resources, typed HTTP APIs, and later source adapters.
///
/// The durable payload boundary is `SourcePayload`, which keeps source-specific
/// IO out of `IndexEngine` while still giving the engine a stable record to
/// extract, chunk, embed, store, and search.
public enum SourceEvent: Codable, Hashable, Sendable {
    case upsert(SourcePayload)
    case delete(documentID: DocumentID)
    case move(documentID: DocumentID, newURI: URL)
    case permissionChanged(documentID: DocumentID)
    /// A path inside the source could not be read during this sync. Nothing under
    /// it was indexed or deleted; indexed documents under it are additionally
    /// reported per-document through `.permissionChanged`.
    case pathUnavailable(path: String, reason: String)
    case sourceUnavailable(SourceID)
    case checkpoint(SourceCursor)
}

public struct ConnectorCapabilities: Codable, Hashable, Sendable {
    public var supportsIncrementalSync: Bool
    public var supportsRuntimeTools: Bool

    public init(supportsIncrementalSync: Bool = true, supportsRuntimeTools: Bool = false) {
        self.supportsIncrementalSync = supportsIncrementalSync
        self.supportsRuntimeTools = supportsRuntimeTools
    }
}

/// Source adapter contract for deterministic ingestion.
///
/// Concrete connectors own authentication, MCP or HTTP transport, retries, rate
/// limits, and source cursors. They do not own chunking, embedding, ranking, or
/// durable index storage.
public protocol SourceConnector: Sendable {
    var id: ConnectorID { get }
    var capabilities: ConnectorCapabilities { get }

    func validate() async throws
    func changes(since cursor: SourceCursor?) async throws -> AsyncThrowingStream<SourceEvent, Error>
    func fetch(_ reference: SourceReference) async throws -> SourcePayload
}

public struct SourceReference: Codable, Hashable, Sendable {
    public var connectorID: ConnectorID
    public var uri: URL
    public var externalID: String?

    public init(connectorID: ConnectorID, uri: URL, externalID: String? = nil) {
        self.connectorID = connectorID
        self.uri = uri
        self.externalID = externalID
    }
}

