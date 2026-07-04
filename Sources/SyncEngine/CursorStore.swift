import ConnectorEngine
import Foundation

/// Durable storage for connector sync cursors, keyed by source. The orchestrator reads the
/// cursor before requesting changes and writes it back only once every emitted mutation has
/// been durably accepted by the engine.
public protocol CursorStore: Sendable {
    func cursor(forKey key: String) -> SourceCursor?
    func setCursor(_ cursor: SourceCursor, forKey key: String)
}

/// Persists all cursors as one dictionary under a single defaults key. UserDefaults is
/// internally thread-safe, hence the unchecked conformance.
public final class UserDefaultsCursorStore: CursorStore, @unchecked Sendable {
    private let defaults: UserDefaults
    private let storageKey: String

    public init(defaults: UserDefaults = .standard, storageKey: String) {
        self.defaults = defaults
        self.storageKey = storageKey
    }

    public func cursor(forKey key: String) -> SourceCursor? {
        (defaults.dictionary(forKey: storageKey) as? [String: SourceCursor])?[key]
    }

    public func setCursor(_ cursor: SourceCursor, forKey key: String) {
        var cursors = defaults.dictionary(forKey: storageKey) as? [String: SourceCursor] ?? [:]
        cursors[key] = cursor
        defaults.set(cursors, forKey: storageKey)
    }
}
