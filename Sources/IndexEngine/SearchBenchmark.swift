import Foundation

/// Runs a fixed query set through an engine and reports per-layer p50/p95 latencies from the
/// search diagnostics — the numbers a retrieval harness exists to answer.
public enum SearchBenchmark {
    /// One search's per-layer latencies, lifted from the engine's diagnostics.
    public struct Sample: Sendable {
        public let sql: TimeInterval?
        public let fts: TimeInterval?
        public let vector: TimeInterval?
        public let fusion: TimeInterval?
        public let total: TimeInterval?

        public init(diagnostics: SearchDiagnostics) {
            sql = diagnostics.sqlFilterLatency
            fts = diagnostics.ftsLatency
            vector = diagnostics.vectorLatency
            fusion = diagnostics.fusionLatency
            total = diagnostics.totalLatency
        }
    }

    /// p50/p95 per retrieval layer over a fixed query set.
    public struct Report: Sendable {
        public struct LayerStats: Identifiable, Sendable {
            public let id: String
            public let p50: TimeInterval?
            public let p95: TimeInterval?
            public let sampleCount: Int
        }

        public let queries: [String]
        public let iterations: Int
        public let layers: [LayerStats]

        public var totalRuns: Int { queries.count * iterations }

        public init(queries: [String], iterations: Int, samples: [Sample]) {
            self.queries = queries
            self.iterations = iterations
            self.layers = [
                Self.stats("Total", samples.compactMap(\.total)),
                Self.stats("SQL", samples.compactMap(\.sql)),
                Self.stats("FTS", samples.compactMap(\.fts)),
                Self.stats("Vector", samples.compactMap(\.vector)),
                Self.stats("Fusion", samples.compactMap(\.fusion)),
            ]
        }

        private static func stats(_ name: String, _ values: [TimeInterval]) -> LayerStats {
            LayerStats(
                id: name,
                p50: SearchBenchmark.percentile(values, 0.50),
                p95: SearchBenchmark.percentile(values, 0.95),
                sampleCount: values.count
            )
        }
    }

    /// Run every query `iterations` times in diagnostic mode and aggregate the layer latencies.
    public static func run(
        engine: any IndexEngineClient,
        queries: [String],
        iterations: Int = 5,
        limit: Int = 10
    ) async throws -> Report {
        let iterations = max(1, iterations)
        var samples: [Sample] = []
        for _ in 0..<iterations {
            for query in queries {
                let response = try await engine.search(.init(query: query, mode: .diagnostic, limit: limit))
                samples.append(Sample(diagnostics: response.diagnostics))
            }
        }
        return Report(queries: queries, iterations: iterations, samples: samples)
    }

    /// Nearest-rank percentile; nil when there are no samples.
    public static func percentile(_ values: [TimeInterval], _ p: Double) -> TimeInterval? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let rank = Int((p * Double(sorted.count)).rounded(.up)) - 1
        return sorted[max(0, min(rank, sorted.count - 1))]
    }
}
