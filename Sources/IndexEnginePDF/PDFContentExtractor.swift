import Foundation
import IndexEngine
import PDFKit

/// Extracts the embedded text layer from PDF payloads via PDFKit.
///
/// The host app registers this through `IndexEngineConfiguration.extractors`; the engine
/// calls `extract` for binary payloads whose content type is `com.adobe.pdf`, then routes
/// the returned text through the normal text → chunk → embed pipeline.
///
/// This reads the *digital* text layer only. Scanned / image-only PDFs carry no extractable
/// text and surface as a typed extraction failure (`noExtractableText`) rather than a silent
/// filename-only document — OCR (Vision) would be a separate `.ocrText` extractor.
public struct PDFContentExtractor: ContentExtractor {
    public let id: ComponentID = "indexengine.extractor.pdf"
    public let version: String = "1"
    public let supportedContentTypes: Set<String> = ["com.adobe.pdf"]

    public init() {}

    public func extract(_ payload: SourcePayload, options: ExtractionOptions) async throws -> [RepresentationInput] {
        guard case let .binaryReference(url) = payload.body else { return [] }
        guard let document = PDFDocument(url: url) else {
            throw PDFExtractionError.unreadable(url)
        }
        guard !document.isLocked else {
            throw PDFExtractionError.locked(url)
        }

        let pageCount = document.pageCount
        var pages: [String] = []
        pages.reserveCapacity(pageCount)
        for index in 0 ..< pageCount {
            try Task.checkCancellation()
            guard let page = document.page(at: index), let text = page.string else { continue }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { pages.append(trimmed) }
        }

        let text = pages.joined(separator: "\n\n")
        guard !text.isEmpty else {
            throw PDFExtractionError.noExtractableText(url)
        }

        return [
            RepresentationInput(
                kind: .plainText,
                text: text,
                metadata: ["pageCount": .integer(Int64(pageCount))]
            )
        ]
    }
}

public enum PDFExtractionError: ContentExtractionError, CustomStringConvertible {
    case unreadable(URL)
    case locked(URL)
    case noExtractableText(URL)

    /// Deliberately omits the file name: every consumer shows it alongside this sentence (the
    /// failure record carries the document ID), so repeating it here only forces redundant,
    /// wrapping copy. The sentence states the cause, nothing the caller already knows.
    public var description: String {
        switch self {
        case .unreadable:
            return "The PDF couldn’t be opened — it may be damaged."
        case .locked:
            return "The PDF is password-protected and must be unlocked before text can be extracted."
        case .noExtractableText:
            return "No selectable text — this looks like a scanned PDF, which needs OCR to be searchable."
        }
    }
}
