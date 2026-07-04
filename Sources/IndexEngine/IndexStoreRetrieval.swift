import Foundation

private let maxSearchLimit = 1_000

extension IndexStore {
    public func search(
        _ query: String,
        scope: Scope = .global,
        filters: SearchFilters = .init(),
        limit: Int = 10
    ) async throws -> [SearchHit] {
        try await searchDetailed(
            query,
            scope: scope,
            filters: filters,
            limit: limit,
            allowDegradedResults: false
        ).hits
    }

    public func searchDetailed(
        _ query: String,
        scope: Scope = .global,
        filters: SearchFilters = .init(),
        limit: Int = 10,
        allowDegradedResults: Bool = true,
        profile: RetrievalProfile = .fast
    ) async throws -> (hits: [SearchHit], diagnostics: SearchDiagnostics) {
        let started = Date.now
        let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedQuery.isEmpty, limit > 0 else {
            return (
                [],
                SearchDiagnostics(totalLatency: Date.now.timeIntervalSince(started))
            )
        }

        let clampedLimit = min(limit, maxSearchLimit)
        let profile = profile.normalized(returnLimit: clampedLimit)
        let pool = max(clampedLimit * 4, 40)
        let exactLimit = min(pool, 200, profile.maxSnippets)
        let keywordLimit = min(pool, profile.maxFTSCandidates)
        let vectorLimit = min(pool, profile.maxVectorCandidates)
        let rerankLimit = max(clampedLimit, profile.maxRerankCandidates)
        let hardCluster = scope.hardScope ? scope.clusterID : nil

        let sqlStart = Date.now
        let exact = exactLimit > 0
            ? try exactIDs(normalizedQuery, hardCluster: hardCluster, filters: filters, limit: exactLimit)
            : []
        let sqlLatency = Date.now.timeIntervalSince(sqlStart)

        let ftsStart = Date.now
        let keyword = keywordLimit > 0
            ? try keywordIDs(normalizedQuery, hardCluster: hardCluster, filters: filters, limit: keywordLimit)
            : []
        let ftsLatency = Date.now.timeIntervalSince(ftsStart)

        var missingChannels: [RetrievalChannel] = []
        var vector: [String] = []
        var vectorLatency: TimeInterval?

        if vectorLimit > 0 {
            do {
                let qvec = try await embedder.embed(normalizedQuery, kind: .query)
                try validateEmbedding(qvec, kind: .query)
                let vectorStart = Date.now
                vector = try vectorIDs(qvec, hardCluster: hardCluster, filters: filters, limit: vectorLimit)
                vectorLatency = Date.now.timeIntervalSince(vectorStart)
            } catch {
                if allowDegradedResults {
                    missingChannels.append(.vector)
                } else {
                    throw error
                }
            }
        }

        let fusionStart = Date.now
        var fused: [String: Double] = [:]
        var exactRank: [String: Int] = [:]
        var keywordRank: [String: Int] = [:]
        var vectorRank: [String: Int] = [:]

        for (index, id) in exact.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            exactRank[id] = index + 1
        }
        for (index, id) in keyword.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            keywordRank[id] = index + 1
        }
        for (index, id) in vector.enumerated() {
            fused[id, default: 0] += 1 / (Self.reciprocalRankK + Double(index + 1))
            vectorRank[id] = index + 1
        }

        if let clusterID = scope.clusterID, !scope.hardScope, scope.boostInScope != 1 {
            let clusterValues = try clusterMap(Set(fused.keys))
            for id in fused.keys where clusterValues[id] == clusterID {
                if let score = fused[id] {
                    fused[id] = score * scope.boostInScope
                }
            }
        }

        let orderedIDs = fused.sorted { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.key < rhs.key
            }
            return lhs.value > rhs.value
        }
        .prefix(rerankLimit)
        .prefix(clampedLimit)

        var hits: [SearchHit] = []
        hits.reserveCapacity(orderedIDs.count)
        for (chunkID, score) in orderedIDs {
            if let metadata = try meta(chunkID) {
                hits.append(
                    SearchHit(
                        id: chunkID,
                        documentID: metadata.documentID,
                        chunkID: chunkID,
                        type: metadata.contentType,
                        title: metadata.title,
                        snippet: Self.snippet(text: metadata.text, query: normalizedQuery),
                        sourceID: metadata.sourceID,
                        sourceURI: metadata.sourceURI,
                        policyID: metadata.policyID,
                        representationID: metadata.representationID,
                        embeddingSpaceID: metadata.embeddingSpaceID,
                        score: score,
                        exactRank: exactRank[chunkID],
                        keywordRank: keywordRank[chunkID],
                        vectorRank: vectorRank[chunkID]
                    )
                )
            }
        }

        let fusionLatency = Date.now.timeIntervalSince(fusionStart)
        return (
            hits,
            SearchDiagnostics(
                degraded: !missingChannels.isEmpty,
                missingChannels: missingChannels,
                sqlFilterLatency: sqlLatency,
                ftsLatency: ftsLatency,
                vectorLatency: vectorLatency,
                fusionLatency: fusionLatency,
                snippetLatency: nil,
                totalLatency: Date.now.timeIntervalSince(started)
            )
        )
    }

    private func exactIDs(
        _ query: String,
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [String] {
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks.id FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        JOIN embeddings ON embeddings.chunk_id = chunks.id
        WHERE chunks.active = 1 AND \(filter.whereSQL)
          AND (documents.title LIKE ? ESCAPE '\\'
               OR documents.source_uri LIKE ? ESCAPE '\\')
        ORDER BY documents.title ASC, chunks.ordinal ASC LIMIT ?
        """)
        var bindIndex: Int32 = 1
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }
        let pattern = Self.likePattern(for: query)
        statement.bind(bindIndex, pattern)
        bindIndex += 1
        statement.bind(bindIndex, pattern)
        bindIndex += 1
        statement.bind(bindIndex, limit)

        var ids: [String] = []
        while try statement.step() {
            if let id = statement.text(0) {
                ids.append(id)
            }
        }
        return ids
    }

    private func keywordIDs(
        _ query: String,
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [String] {
        let tokens = HashingEmbedder.tokens(query)
        guard !tokens.isEmpty else { return [] }
        let match = tokens.map { "\"\($0)\"" }.joined(separator: " OR ")
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks_fts.chunk_id FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        JOIN embeddings ON embeddings.chunk_id = chunks.id
        WHERE chunks_fts MATCH ? AND chunks.active = 1 AND \(filter.whereSQL)
        ORDER BY bm25(chunks_fts) ASC LIMIT ?
        """)
        statement.bind(1, match)
        var bindIndex: Int32 = 2
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }
        statement.bind(bindIndex, limit)

        var ids: [String] = []
        while try statement.step() {
            if let id = statement.text(0) {
                ids.append(id)
            }
        }
        return ids
    }

    private func vectorIDs(
        _ qvec: [Float],
        hardCluster: String?,
        filters: SearchFilters,
        limit: Int
    ) throws -> [String] {
        let filter = candidateFilterSQL(hardCluster: hardCluster, filters: filters)
        let statement = try db.prepare("""
        SELECT chunks.id, vectors.vec FROM embeddings
        JOIN vectors ON vectors.id = embeddings.id
        JOIN chunks ON chunks.id = embeddings.chunk_id
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.active = 1 AND vectors.dim = ? AND \(filter.whereSQL)
        """)
        statement.bind(1, dimension)
        var bindIndex: Int32 = 2
        for value in filter.bindings {
            statement.bind(bindIndex, value)
            bindIndex += 1
        }

        var scored: [(String, Float)] = []
        while try statement.step() {
            guard let id = statement.text(0) else { continue }
            let vector = Vector.fromBytes(statement.blob(1))
            guard vector.count == dimension else {
                throw IndexStoreError.storedVectorDimensionMismatch(id: id, expected: dimension, actual: vector.count)
            }
            scored.append((id, Vector.cosine(qvec, vector)))
        }
        return scored.sorted { lhs, rhs in
            if lhs.1 == rhs.1 {
                return lhs.0 < rhs.0
            }
            return lhs.1 > rhs.1
        }
        .prefix(limit)
        .map(\.0)
    }

    private func candidateFilterSQL(
        hardCluster: String?,
        filters: SearchFilters
    ) -> CandidateFilter {
        var clauses = [
            "documents.is_deleted = 0",
            "embeddings.embedding_space_id = ?"
        ]
        var bindings = [filters.embeddingSpaceID?.rawValue ?? embeddingSpaceID]

        if let hardCluster {
            clauses.append("documents.cluster_id = ?")
            bindings.append(hardCluster)
        }
        if !filters.sourceIDs.isEmpty {
            let sourceIDs = filters.sourceIDs.map(\.rawValue).sorted()
            clauses.append("documents.source_id IN (\(Self.placeholders(count: sourceIDs.count)))")
            bindings.append(contentsOf: sourceIDs)
        }
        if !filters.contentTypes.isEmpty {
            let contentTypes = filters.contentTypes.sorted()
            clauses.append("documents.content_type IN (\(Self.placeholders(count: contentTypes.count)))")
            bindings.append(contentsOf: contentTypes)
        }
        if let policyID = filters.policyID {
            clauses.append("chunks.policy_id = ?")
            bindings.append(policyID.rawValue)
        }

        return CandidateFilter(whereSQL: clauses.joined(separator: " AND "), bindings: bindings)
    }

    private func clusterMap(_ ids: Set<String>) throws -> [String: String] {
        var map: [String: String] = [:]
        let statement = try db.prepare("""
        SELECT documents.cluster_id FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.id = ?1
        """)
        for id in ids {
            statement.reset()
            statement.bind(1, id)
            if try statement.step(), let clusterID = statement.text(0), !clusterID.isEmpty {
                map[id] = clusterID
            }
        }
        return map
    }

    private func meta(_ chunkID: String) throws -> (
        documentID: String,
        contentType: String,
        title: String,
        text: String,
        sourceID: String?,
        sourceURI: URL?,
        policyID: String?,
        representationID: String?,
        embeddingSpaceID: String?
    )? {
        let statement = try db.prepare("""
        SELECT chunks.document_id,documents.content_type,documents.title,chunks.text,
               documents.source_id,documents.source_uri,chunks.policy_id,chunks.representation_id,
               embeddings.embedding_space_id
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        LEFT JOIN embeddings ON embeddings.chunk_id = chunks.id AND embeddings.embedding_space_id = ?2
        WHERE chunks.id = ?1 AND chunks.active = 1
        LIMIT 1
        """)
        statement.bind(1, chunkID)
        statement.bind(2, embeddingSpaceID)
        guard try statement.step() else { return nil }
        let sourceURI = statement.text(5).flatMap(URL.init(string:))
        return (
            statement.text(0) ?? "",
            statement.text(1) ?? "",
            statement.text(2) ?? "",
            statement.text(3) ?? "",
            emptyStringAsNil(statement.text(4)),
            sourceURI,
            emptyStringAsNil(statement.text(6)),
            emptyStringAsNil(statement.text(7)),
            emptyStringAsNil(statement.text(8))
        )
    }

    private static func snippet(text: String, query: String) -> String {
        let limit = 240
        guard text.count > limit else { return text }
        let tokens = HashingEmbedder.tokens(query)
        let matchRange = tokens.compactMap { token in
            text.range(of: token, options: [.caseInsensitive, .diacriticInsensitive])
        }.first
        let center = matchRange.map { text.distance(from: text.startIndex, to: $0.lowerBound) } ?? 0
        let startOffset = max(0, center - 80)
        let start = text.index(text.startIndex, offsetBy: startOffset, limitedBy: text.endIndex) ?? text.startIndex
        let end = text.index(start, offsetBy: limit, limitedBy: text.endIndex) ?? text.endIndex
        let prefix = start == text.startIndex ? "" : "..."
        let suffix = end == text.endIndex ? "" : "..."
        return prefix + text[start..<end].trimmingCharacters(in: .whitespacesAndNewlines) + suffix
    }
}
