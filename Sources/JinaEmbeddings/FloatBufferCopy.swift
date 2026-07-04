import Foundation

enum FloatBufferCopyError: Error, Equatable {
    case countExceedsSource(requested: Int, available: Int)
}

/// Raw little-endian float32 file -> [Float]; shared by every resource loader.
func loadF32(_ url: URL) throws -> [Float] {
    try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

func copyFloats(_ source: [Float], to destination: UnsafeMutablePointer<Float>, count: Int) throws {
    guard count <= source.count else {
        throw FloatBufferCopyError.countExceedsSource(requested: count, available: source.count)
    }
    guard count > 0 else { return }

    try source.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else {
            throw FloatBufferCopyError.countExceedsSource(requested: count, available: 0)
        }
        destination.update(from: baseAddress, count: count)
    }
}
