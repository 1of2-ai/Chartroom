import Testing
@testable import JinaEmbeddings

// `copyFloats` is the one bounds-checked primitive every CoreML tensor pack routes through (the image,
// audio, and video encoders and decoders). It replaced raw `p.update(from: $0.baseAddress!, count:)`
// copies — removing a force-unwrap that crashed on empty input and adding a source-overflow guard.
// These pin that contract so the safety guarantee cannot silently regress into a read past the source.

@Test func copyFloatsCopiesTheExactCount() throws {
    let source: [Float] = [1, 2, 3, 4]
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: 4)
    defer { dst.deallocate() }
    dst.initialize(repeating: -999, count: 4)
    try copyFloats(source, to: dst, count: 4)
    #expect(Array(UnsafeBufferPointer(start: dst, count: 4)) == [1, 2, 3, 4])
}

@Test func copyFloatsCopiesOnlyThePrefixWhenCountIsSmaller() throws {
    let source: [Float] = [1, 2, 3, 4]
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: 4)
    defer { dst.deallocate() }
    dst.initialize(repeating: -999, count: 4)
    try copyFloats(source, to: dst, count: 2)
    #expect(Array(UnsafeBufferPointer(start: dst, count: 4)) == [1, 2, -999, -999])
}

@Test func copyFloatsThrowsRatherThanReadingPastTheSource() {
    let source: [Float] = [1, 2]
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: 4)
    defer { dst.deallocate() }
    dst.initialize(repeating: -999, count: 4)
    #expect(throws: FloatBufferCopyError.countExceedsSource(requested: 4, available: 2)) {
        try copyFloats(source, to: dst, count: 4)
    }
    // Destination must be untouched — the guard fires before any write, so there is no partial or
    // overflowing copy left behind.
    #expect(Array(UnsafeBufferPointer(start: dst, count: 4)) == [-999, -999, -999, -999])
}

@Test func copyFloatsTreatsZeroCountAsANoOpEvenForAnEmptySource() throws {
    // The pre-extraction code force-unwrapped baseAddress and would crash here; this pins that it cannot.
    let dst = UnsafeMutablePointer<Float>.allocate(capacity: 2)
    defer { dst.deallocate() }
    dst.initialize(repeating: -999, count: 2)
    try copyFloats([], to: dst, count: 0)
    #expect(Array(UnsafeBufferPointer(start: dst, count: 2)) == [-999, -999])
}
