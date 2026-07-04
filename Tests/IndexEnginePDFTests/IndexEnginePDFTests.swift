import AppKit
import CoreGraphics
import CoreText
import Foundation
import IndexEngine
import PDFKit
import Testing
@testable import IndexEnginePDF

@Suite("IndexEnginePDF — PDF content extraction")
struct IndexEnginePDFTests {
    @Test("extracts the embedded text layer with page count")
    func extractsTextLayer() async throws {
        let url = try Self.makeTextPDF("thermal throttling under sustained load")
        defer { try? FileManager.default.removeItem(at: url) }

        let payload = SourcePayload(
            documentID: "doc-pdf",
            displayName: "doc.pdf",
            contentType: "com.adobe.pdf",
            body: .binaryReference(url)
        )
        let reps = try await PDFContentExtractor().extract(payload, options: ExtractionOptions())
        let rep = try #require(reps.first)

        #expect(rep.kind == .plainText)
        #expect(rep.text.contains("throttling"))
        #expect(rep.metadata["pageCount"] == .integer(1))
    }

    @Test("a non-PDF file throws a typed extraction error")
    func unreadableThrows() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pdf")
        try Data("not a pdf".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let payload = SourcePayload(
            documentID: "doc-bad",
            displayName: "bad.pdf",
            contentType: "com.adobe.pdf",
            body: .binaryReference(url)
        )
        await #expect(throws: PDFExtractionError.self) {
            _ = try await PDFContentExtractor().extract(payload, options: ExtractionOptions())
        }
    }

    @Test("a password-protected PDF is diagnosed as locked, not scanned")
    func lockedPDFThrowsLockedError() async throws {
        let url = try Self.makeLockedPDF("secret text layer")
        defer { try? FileManager.default.removeItem(at: url) }

        let payload = SourcePayload(
            documentID: "doc-locked",
            displayName: "locked.pdf",
            contentType: "com.adobe.pdf",
            body: .binaryReference(url)
        )

        do {
            _ = try await PDFContentExtractor().extract(payload, options: ExtractionOptions())
            Issue.record("Expected a locked PDF error")
        } catch PDFExtractionError.locked(let lockedURL) {
            #expect(lockedURL == url)
        } catch {
            Issue.record("Expected locked PDF error, got \(error)")
        }
    }

    @Test("end to end: a PDF ingested through the engine becomes searchable by its content")
    func endToEndIngestAndSearch() async throws {
        let url = try Self.makeTextPDF("thermal throttling under sustained load")
        defer { try? FileManager.default.removeItem(at: url) }

        // Mock embedder keeps this fast and offline; the point is that the extractor is
        // invoked during ingest and the PDF's text (not just its filename) is indexed.
        let engine = try await IndexEngine.openInMemory(
            configuration: IndexEngineConfiguration(
                embedder: HashingEmbedder(),
                extractors: [PDFContentExtractor()]
            )
        )

        _ = try await engine.ingest(IngestRequest(payloads: [
            SourcePayload(
                documentID: "doc-pdf",
                displayName: "doc.pdf",
                contentType: "com.adobe.pdf",
                body: .binaryReference(url)
            )
        ]))

        // "throttling" appears only inside the PDF content, never in the filename.
        let response = try await engine.search(SearchRequest(query: "throttling"))
        let top = try #require(response.results.first)
        #expect(top.documentID == "doc-pdf")
    }

    /// Render a single-page PDF whose text layer is `text`, via Core Text into a PDF
    /// `CGContext`. PDFKit can extract the result, exercising the real path.
    static func makeTextPDF(_ text: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pdf")
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else {
            throw CocoaError(.fileWriteUnknown)
        }
        context.beginPDFPage(nil)
        let attributed = NSAttributedString(
            string: text,
            attributes: [.font: NSFont.systemFont(ofSize: 24)]
        )
        let line = CTLineCreateWithAttributedString(attributed)
        context.textPosition = CGPoint(x: 72, y: 700)
        CTLineDraw(line, context)
        context.endPDFPage()
        context.closePDF()
        return url
    }

    static func makeLockedPDF(_ text: String) throws -> URL {
        let unlockedURL = try makeTextPDF(text)
        defer { try? FileManager.default.removeItem(at: unlockedURL) }
        let lockedURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pdf")
        let document = try #require(PDFDocument(url: unlockedURL))
        let options: [PDFDocumentWriteOption: Any] = [
            .ownerPasswordOption: "owner-password",
            .userPasswordOption: "user-password"
        ]
        let wrote = document.write(
            to: lockedURL,
            withOptions: options
        )
        guard wrote else { throw CocoaError(.fileWriteUnknown) }
        return lockedURL
    }
}
