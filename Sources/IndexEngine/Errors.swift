import Foundation
import UniformTypeIdentifiers

/// Typed error surface for applications and GUI code.
public struct IndexEngineError: Error, Sendable, CustomStringConvertible {
    public enum Category: String, Codable, Sendable {
        case configurationInvalid
        case registryUnsatisfied
        case policyQuarantined
        case migrationRequired
        case migrationFailed
        case sourceRefetchRequired
        case embeddingProviderUnavailable
        case embeddingSpaceUnavailable
        case vectorBackendUnavailable
        case storageUnavailable
        case ingestionFailed
        case deletionFailed
        case searchDegraded
        case permissionDenied
        case cancelled
    }

    public enum Recoverability: String, Codable, Hashable, Sendable {
        case retryable
        case needsUserAction
        case needsConfiguration
        case unrecoverable
    }

    public var category: Category
    public var code: String
    public var recoverability: Recoverability
    public var summary: String
    public var detail: String
    public var relatedIDs: [EngineID]

    public init(
        _ category: Category,
        code: String,
        recoverability: Recoverability,
        summary: String,
        detail: String = "",
        relatedIDs: [EngineID] = []
    ) {
        self.category = category
        self.code = code
        self.recoverability = recoverability
        self.summary = summary
        self.detail = detail
        self.relatedIDs = relatedIDs
    }

    public var description: String {
        detail.isEmpty ? "\(code): \(summary)" : "\(code): \(summary). \(detail)"
    }
}

/// The one contentType → RepresentationKind mapping for the module. The store and the
/// payload extension previously carried diverging copies (their fallbacks disagreed on
/// MIME-typed JSON, which has no UTType).
func resolveRepresentationKind(forContentType contentType: String) -> RepresentationKind {
    if contentType == "net.daringfireball.markdown" {
        return .markdown
    }
    guard let type = UTType(contentType) else {
        return contentType.contains("json") ? .structuredJSON : .plainText
    }
    if type == .json {
        return .structuredJSON
    }
    if type.conforms(to: .sourceCode) {
        return .code
    }
    return .plainText
}

enum PayloadExtractionError: Error, CustomStringConvertible {
    case unsupportedReference(URL)
    case unreadableTextReference(URL, String)

    var description: String {
        switch self {
        case let .unsupportedReference(url):
            "Unsupported binary reference: \(url.absoluteString)"
        case let .unreadableTextReference(url, reason):
            "Could not read text reference \(url.path): \(reason)"
        }
    }
}

extension FailureSnapshot.Category {
    /// Classify an arbitrary error into a failure category. Public so connector-side
    /// ingestion helpers record failures with the same taxonomy as the engine.
    public init(_ error: Error) {
        if error is PayloadExtractionError || error is ContentExtractionError {
            self = .extractionFailure
        } else if error is IndexStoreError {
            self = .embeddingFailure
        } else if error is SQLiteError {
            self = .storageFailure
        } else {
            self = .embeddingFailure
        }
    }
}

extension IndexEngineError.Recoverability {
    /// Classify an arbitrary error's recoverability; see `FailureSnapshot.Category.init(_:)`.
    public init(_ error: Error) {
        switch error {
        case let engineError as IndexEngineError:
            self = engineError.recoverability
        case is PayloadExtractionError:
            self = .needsUserAction
        case is IndexStoreError:
            self = .needsConfiguration
        case is SQLiteError:
            self = .retryable
        default:
            self = .needsConfiguration
        }
    }
}
