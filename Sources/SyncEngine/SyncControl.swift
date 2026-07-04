import Foundation

/// Pause / resume / stop control for one sync run. The requesting side (typically a UI) calls
/// the synchronous state transitions; the orchestrator's ingest loop suspends on
/// `waitWhilePaused()` between payloads and checks `isStopping` at each checkpoint.
///
/// Lock-based rather than an actor so callers can flip state synchronously — a pause must be
/// observable by the loop's very next checkpoint without an actor-hop race.
public final class SyncControl: @unchecked Sendable {
    public enum State: Sendable {
        case running, paused, stopping
    }

    private let lock = NSLock()
    private var _state: State = .running
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() {}

    public var state: State {
        lock.withLock { _state }
    }

    public func pause() {
        lock.withLock {
            if _state == .running { _state = .paused }
        }
    }

    public func resume() {
        let resumed: [CheckedContinuation<Void, Never>] = lock.withLock {
            guard _state == .paused else { return [] }
            _state = .running
            defer { waiters = [] }
            return waiters
        }
        for continuation in resumed { continuation.resume() }
    }

    public func stop() {
        let resumed: [CheckedContinuation<Void, Never>] = lock.withLock {
            guard _state != .stopping else { return [] }
            _state = .stopping
            defer { waiters = [] }
            return waiters
        }
        for continuation in resumed { continuation.resume() }
    }

    var isStopping: Bool {
        state == .stopping
    }

    func waitWhilePaused() async {
        while true {
            guard lock.withLock({ _state == .paused }) else { return }
            await withCheckedContinuation { continuation in
                let resumeImmediately: Bool = lock.withLock {
                    guard _state == .paused else { return true }
                    waiters.append(continuation)
                    return false
                }
                if resumeImmediately { continuation.resume() }
            }
        }
    }
}
