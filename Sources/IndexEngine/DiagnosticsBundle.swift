import Foundation

/// An engine diagnostic bundle — every snapshot a client reads, as pretty-printed JSON files
/// in a directory. Hosts decide how to package the directory (zip, tar, share sheet).
public struct DiagnosticsBundle: Sendable {
    public var snapshot: IndexEngineSnapshot
    public var health: IndexHealthSnapshot
    public var modelStatus: ModelStatusSnapshot
    public var jobs: [JobSnapshot]
    public var failures: [FailureSnapshot]
    public var lastSearch: SearchResponse?

    public init(
        snapshot: IndexEngineSnapshot,
        health: IndexHealthSnapshot,
        modelStatus: ModelStatusSnapshot,
        jobs: [JobSnapshot],
        failures: [FailureSnapshot],
        lastSearch: SearchResponse? = nil
    ) {
        self.snapshot = snapshot
        self.health = health
        self.modelStatus = modelStatus
        self.jobs = jobs
        self.failures = failures
        self.lastSearch = lastSearch
    }

    /// Capture a fresh bundle from the engine. `lastSearch` is passed in because the engine
    /// does not retain search responses; the client owns the most recent one.
    public static func capture(
        from engine: any IndexEngineClient,
        lastSearch: SearchResponse? = nil,
        limit: Int
    ) async -> DiagnosticsBundle {
        await DiagnosticsBundle(
            snapshot: engine.snapshot(),
            health: engine.health(),
            modelStatus: engine.modelStatus(),
            jobs: engine.jobs(limit: limit),
            failures: engine.failures(limit: limit),
            lastSearch: lastSearch
        )
    }

    /// Write the bundle's JSON files into `directory` (which must already exist).
    public func write(to directory: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        func write(_ value: some Encodable, as name: String) throws {
            try encoder.encode(value).write(to: directory.appendingPathComponent(name))
        }

        try write(snapshot, as: "snapshot.json")
        try write(health, as: "health.json")
        try write(modelStatus, as: "model-status.json")
        try write(jobs, as: "jobs.json")
        try write(failures, as: "failures.json")
        if let lastSearch {
            try write(lastSearch, as: "last-search.json")
        }
    }
}
