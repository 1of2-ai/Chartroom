import CoreML

/// Guarded raw access to `MLMultiArray` contents (replaces the deprecated `dataPointer`).
///
/// Model OUTPUT tensors are re-checked at runtime: a re-exported bundle whose tensors come back
/// float16 — or with padded (non-dense) strides — would otherwise be reinterpreted as packed
/// float32 and silently produce garbage vectors. Host-allocated INPUT tensors have a caller-chosen
/// dtype, so only their layout is re-checked before linear indexing.
enum TensorAccessError: Error, Equatable, CustomStringConvertible {
    case unexpectedTensorType(expected: MLMultiArrayDataType, actual: MLMultiArrayDataType)
    case nonDenseLayout(shape: [Int], strides: [Int])

    var description: String {
        switch self {
        case .unexpectedTensorType(let expected, let actual):
            return "unexpected tensor dtype \(name(of: actual)) (expected \(name(of: expected)))"
        case .nonDenseLayout(let shape, let strides):
            return "non-dense tensor layout (shape \(shape), strides \(strides))"
        }
    }

    private func name(of type: MLMultiArrayDataType) -> String {
        // Older SDKs do not expose the .int8 case even though the raw dtype value is stable.
        if type.rawValue == 131_080 {
            return "int8"
        }

        switch type {
        case .double, .float64: return "float64"
        case .float32: return "float32"
        case .float16: return "float16"
        case .int32: return "int32"
        default: return "raw(\(type.rawValue))"
        }
    }
}

/// True when `strides` describe the packed row-major layout linear indexing assumes.
/// (Size-1 axes never affect addressing, so their stride value is ignored.)
private func isDense(shape: [NSNumber], strides: [NSNumber]) -> Bool {
    guard shape.count == strides.count else { return false }
    var expected = 1
    for i in stride(from: shape.count - 1, through: 0, by: -1) {
        let dim = shape[i].intValue
        if dim != 1, strides[i].intValue != expected { return false }
        expected *= dim
    }
    return true
}

private func requireDense(_ array: MLMultiArray) throws {
    guard isDense(shape: array.shape, strides: array.strides) else {
        throw TensorAccessError.nonDenseLayout(shape: array.shape.map(\.intValue),
                                               strides: array.strides.map(\.intValue))
    }
}

/// Copy a float32 model-output tensor into a flat `[Float]`, guarding dtype + dense layout.
func floatValues(of array: MLMultiArray) throws -> [Float] {
    try withFloat32Tensor(array) { Array($0.prefix(array.count)) }
}

/// Read a float32 model-output tensor in place (no flat copy), guarding dtype + dense layout.
func withFloat32Tensor<R>(_ array: MLMultiArray, _ body: (UnsafeBufferPointer<Float>) throws -> R) throws -> R {
    guard array.dataType == .float32 else {
        throw TensorAccessError.unexpectedTensorType(expected: .float32, actual: array.dataType)
    }
    try requireDense(array)
    return try array.withUnsafeBufferPointer(ofType: Float.self, body)
}

/// Write into a host-allocated tensor through a dense buffer (the dtype was chosen by the caller
/// at allocation, so only the layout is re-checked before linear indexing).
func withDenseTensor<S: MLShapedArrayScalar, R>(
    _ array: MLMultiArray, as type: S.Type,
    _ body: (UnsafeMutableBufferPointer<S>) throws -> R
) throws -> R {
    try requireDense(array)
    return try array.withUnsafeMutableBufferPointer(ofType: type) { ptr, _ in try body(ptr) }
}
