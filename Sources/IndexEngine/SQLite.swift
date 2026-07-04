import Foundation
import SQLite3

/// SQLite wants the TRANSIENT destructor (it copies the bound bytes during the
/// call). It is not exported to Swift, so reconstruct it.
@inline(__always) private func sqliteTransient() -> sqlite3_destructor_type {
    unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}

public struct SQLiteError: Error, CustomStringConvertible {
    public let code: Int32
    public let message: String
    public var description: String { "SQLite error \(code): \(message)" }
}

/// A thin, focused wrapper over the system SQLite C API. Used only inside the
/// `IndexStore` actor, so it does not need to be Sendable.
final class SQLite {
    let handle: OpaquePointer
    let path: String

    init(path: String) throws {
        self.path = path
        var h: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(path, &h, flags, nil)
        guard rc == SQLITE_OK, let h else {
            let msg = h.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            if let h { sqlite3_close_v2(h) }
            throw SQLiteError(code: rc, message: msg)
        }
        handle = h
        let timeoutRC = sqlite3_busy_timeout(handle, 5_000)
        guard timeoutRC == SQLITE_OK else {
            throw SQLiteError(code: timeoutRC, message: String(cString: sqlite3_errmsg(handle)))
        }
        try exec("PRAGMA journal_mode=WAL")
        try exec("PRAGMA foreign_keys=ON")
    }

    deinit { sqlite3_close_v2(handle) }

    /// Total on-disk footprint of the database and its WAL/SHM sidecars in bytes, or
    /// nil for an in-memory store. WAL mode (enabled above) keeps `-wal` and `-shm`
    /// companion files next to the main database; that layout is this wrapper's concern,
    /// so callers ask the store for its size rather than knowing the file naming.
    var fileByteSize: Int64? {
        guard path != ":memory:", !path.isEmpty else { return nil }
        let total = [path, path + "-wal", path + "-shm"].reduce(Int64(0)) { sum, candidate in
            let size = (try? FileManager.default.attributesOfItem(atPath: candidate))?[.size] as? Int64 ?? 0
            return sum + size
        }
        return total > 0 ? total : nil
    }

    func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let m = err.map { String(cString: $0) } ?? String(cString: sqlite3_errmsg(handle))
            sqlite3_free(err)
            throw SQLiteError(code: rc, message: m)
        }
    }

    func transaction<T>(_ body: () throws -> T) throws -> T {
        try exec("BEGIN IMMEDIATE TRANSACTION")
        do {
            let result = try body()
            try exec("COMMIT")
            return result
        } catch {
            try? exec("ROLLBACK")
            throw error
        }
    }

    func prepare(_ sql: String) throws -> Statement {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(handle, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK, let stmt else {
            throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(handle)))
        }
        return Statement(stmt: stmt, db: handle)
    }
}

final class Statement {
    private let stmt: OpaquePointer
    private let db: OpaquePointer

    init(stmt: OpaquePointer, db: OpaquePointer) { self.stmt = stmt; self.db = db }
    deinit { sqlite3_finalize(stmt) }

    func bind(_ i: Int32, _ text: String) { checkBind(sqlite3_bind_text(stmt, i, text, -1, sqliteTransient())) }
    func bind(_ i: Int32, _ value: Double) { checkBind(sqlite3_bind_double(stmt, i, value)) }
    func bind(_ i: Int32, _ value: Int) { checkBind(sqlite3_bind_int64(stmt, i, Int64(value))) }
    func bindNull(_ i: Int32) { checkBind(sqlite3_bind_null(stmt, i)) }
    func bindBlob(_ i: Int32, _ bytes: [UInt8]) {
        bytes.withUnsafeBytes { checkBind(sqlite3_bind_blob(stmt, i, $0.baseAddress, Int32($0.count), sqliteTransient())) }
    }

    private func checkBind(_ rc: Int32) {
        precondition(rc == SQLITE_OK, String(cString: sqlite3_errmsg(db)))
    }

    /// Rewind for reuse: clears the row cursor and bindings so one prepared
    /// statement can run repeatedly inside a loop instead of re-preparing.
    func reset() {
        sqlite3_reset(stmt)
        sqlite3_clear_bindings(stmt)
    }

    /// True if a row is available, false when the statement is done.
    @discardableResult func step() throws -> Bool {
        let rc = sqlite3_step(stmt)
        if rc == SQLITE_ROW { return true }
        if rc == SQLITE_DONE { return false }
        throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(db)))
    }

    func text(_ col: Int32) -> String? {
        guard let c = sqlite3_column_text(stmt, col) else { return nil }
        return String(cString: c)
    }
    func int(_ col: Int32) -> Int { Int(sqlite3_column_int64(stmt, col)) }
    func double(_ col: Int32) -> Double { sqlite3_column_double(stmt, col) }
    func null(_ col: Int32) -> Bool { sqlite3_column_type(stmt, col) == SQLITE_NULL }
    func blob(_ col: Int32) -> [UInt8] {
        guard let p = sqlite3_column_blob(stmt, col) else { return [] }
        let n = Int(sqlite3_column_bytes(stmt, col))
        return Array(UnsafeRawBufferPointer(start: p, count: n))
    }
}

/// Float vector <-> bytes (host-endian; the DB is local to one machine) and the
/// similarity used for ranking. Embeddings are L2-normalized, so cosine = dot.
enum Vector {
    static func toBytes(_ v: [Float]) -> [UInt8] { v.withUnsafeBytes { Array($0) } }

    static func fromBytes(_ b: [UInt8]) -> [Float] {
        let n = b.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBytes { dst in
            b.withUnsafeBytes { src in
                if let base = src.baseAddress { dst.copyMemory(from: UnsafeRawBufferPointer(start: base, count: n * MemoryLayout<Float>.size)) }
            }
        }
        return out
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<a.count { s += a[i] * b[i] }
        return s
    }
}
