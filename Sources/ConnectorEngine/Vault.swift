import Foundation
import IndexEngine

/// Parses Obsidian-style markdown into the index's search projection. The `.md`
/// on disk stays canonical (the Obsidian exception); what we store is metadata +
/// body text for retrieval, keyed by the vault-relative path so a hit points back
/// at the file.
public enum Vault {
    /// Frontmatter (`--- ... ---`) supplies id/type/title/cluster when present;
    /// otherwise they are derived (id = relative path, title = first `#` heading
    /// or the file stem, type = "note").
    public static func parse(markdown text: String, relativePath: String) -> IndexedObject {
        let (front, body) = splitFrontmatter(text)
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        return IndexedObject(
            id: front["id"] ?? relativePath,
            type: front["type"] ?? "note",
            title: front["title"] ?? firstHeading(body) ?? fileStem(relativePath),
            body: trimmedBody,
            clusterID: front["cluster"]
        )
    }

    static func splitFrontmatter(_ text: String) -> (front: [String: String], body: String) {
        let lines = text.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---" else { return ([:], text) }
        var front: [String: String] = [:]
        var i = 1
        var foundClosingDelimiter = false
        while i < lines.count {
            let line = lines[i]
            i += 1
            if line.trimmingCharacters(in: .whitespaces) == "---" {
                foundClosingDelimiter = true
                break
            }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces)
            let value = String(line[line.index(after: colon)...])
                .trimmingCharacters(in: .whitespaces)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            if !key.isEmpty { front[key] = value }
        }
        guard foundClosingDelimiter else { return ([:], text) }
        return (front, lines[i...].joined(separator: "\n"))
    }

    static func firstHeading(_ body: String) -> String? {
        for line in body.components(separatedBy: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("# ") { return String(t.dropFirst(2)).trimmingCharacters(in: .whitespaces) }
        }
        return nil
    }

    static func fileStem(_ path: String) -> String {
        (((path as NSString).lastPathComponent) as NSString).deletingPathExtension
    }

    /// All `.md` files under `directory`, gathered synchronously (the directory
    /// enumerator's iterator is unavailable from async contexts).
    static func markdownFiles(in directory: URL) -> [URL] {
        guard let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else { return [] }
        var out: [URL] = []
        while let object = walker.nextObject() {
            if let url = object as? URL, url.pathExtension.lowercased() == "md" { out.append(url) }
        }
        return out
    }
}

public extension IndexStore {
    /// Ingest every `.md` file under `directory` (an Obsidian vault) into the
    /// index. The markdown stays canonical on disk; the stored Object is a search
    /// projection keyed by the vault-relative path. Returns the count ingested.
    ///
    /// Note: file IO runs on the actor here for simplicity; a later pass can read
    /// off-actor and batch the upserts.
    @discardableResult
    func ingestVault(at directory: URL) async throws -> Int {
        let base = directory.standardizedFileURL.path
        var count = 0
        for url in Vault.markdownFiles(in: directory) {
            let rel = Vault.relativePath(for: url, basePath: base)
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                try recordFailure(
                    FailureSnapshot(
                        id: EngineID(rawValue: UUID().uuidString),
                        category: .extractionFailure,
                        message: "Could not ingest vault file \(rel)",
                        detail: String(describing: error),
                        documentID: EngineID(rawValue: rel),
                        recoverability: .needsUserAction,
                        occurredAt: Date.now
                    )
                )
                continue
            }

            do {
                try await upsert(Vault.parse(markdown: text, relativePath: rel))
                count += 1
            } catch {
                try recordFailure(
                    FailureSnapshot(
                        id: EngineID(rawValue: UUID().uuidString),
                        category: FailureSnapshot.Category(error),
                        message: "Could not ingest vault file \(rel)",
                        detail: String(describing: error),
                        documentID: EngineID(rawValue: rel),
                        recoverability: IndexEngineError.Recoverability(error),
                        occurredAt: Date.now
                    )
                )
            }
        }
        return count
    }
}

private extension Vault {
    static func relativePath(for url: URL, basePath: String) -> String {
        var rel = url.standardizedFileURL.path
        if rel.hasPrefix(basePath + "/") {
            rel.removeFirst(basePath.count + 1)
        }
        return rel
    }
}
