import Foundation
import IndexEngine
import UniformTypeIdentifiers

/// Rebuilds retryable local-file payloads from engine failure records. Only failures whose
/// source file still exists on disk can be retried; everything else is skipped.
public enum LocalFileRetry {
    /// The source ID `LocalFileConnector`-backed ingests record on their payloads.
    public static let localFilesSourceID: SourceID = "local-files"

    public static func payloads(
        for failures: [FailureSnapshot],
        sourceID: SourceID = LocalFileRetry.localFilesSourceID
    ) -> [SourcePayload] {
        failures.compactMap { failure in
            guard failure.sourceID == sourceID, let documentID = failure.documentID else { return nil }
            return payload(forDocumentID: documentID, sourceURI: failure.sourceURI, sourceID: sourceID)
        }
    }

    /// Rebuild one payload from the failure's recorded source location.
    static func payload(forDocumentID documentID: DocumentID, sourceURI: URL?, sourceID: SourceID) -> SourcePayload? {
        guard let fileURL = sourceURI?.standardizedFileURL, fileURL.isFileURL,
              FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        let contentType = (try? fileURL.resourceValues(forKeys: [.contentTypeKey]).contentType)?.identifier
            ?? UTType.data.identifier
        return SourcePayload(
            documentID: documentID,
            sourceID: sourceID,
            sourceURI: fileURL,
            displayName: fileURL.lastPathComponent,
            contentType: contentType,
            body: .binaryReference(fileURL)
        )
    }
}
