import Accelerate
import Foundation

/// Whisper-style log-mel front-end (feature_size=128, n_fft=400, hop=160), matching
/// transformers WhisperFeatureExtractor. n_fft=400 is not a power of two, so the DFT is done as
/// a matmul against precomputed cos/sin bases (Accelerate BLAS) rather than an FFT.
///
/// `packedMel(audio)` returns the packed mel `(nMels=128, nFrames)` row-major — the encoder input.
public struct JinaMelFrontend {
    public let nFFT = 400, hop = 160, nMels = 128, nFreq = 201
    public let nFrames: Int
    let window: [Float]       // (400,)
    let melFilters: [Float]   // (201,128) row-major
    let cosMat: [Float]       // (400,201) row-major
    let sinMat: [Float]

    public enum MelError: Error { case missingBundledResource, badResource, audioTooShort(Int) }

    /// Load the mel filterbank + window bundled with the package (no external files needed).
    public init(nFrames: Int = 200) throws {
        guard let mf = Bundle.module.url(forResource: "mel_filters", withExtension: "f32"),
              let w = Bundle.module.url(forResource: "mel_window", withExtension: "f32") else {
            throw MelError.missingBundledResource
        }
        try self.init(melFiltersURL: mf, windowURL: w, nFrames: nFrames)
    }

    public init(melFiltersURL: URL, windowURL: URL, nFrames: Int = 200) throws {
        self.nFrames = nFrames
        melFilters = try loadF32(melFiltersURL)
        window = try loadF32(windowURL)
        guard melFilters.count == nFreq * nMels, window.count == nFFT else { throw MelError.badResource }
        var c = [Float](repeating: 0, count: nFFT * nFreq)
        var s = [Float](repeating: 0, count: nFFT * nFreq)
        for n in 0..<nFFT {
            for k in 0..<nFreq {
                let a = 2.0 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                c[n * nFreq + k] = cos(a); s[n * nFreq + k] = sin(a)
            }
        }
        cosMat = c; sinMat = s
    }

    /// Row-major C(M,N) = A(M,K) @ B(K,N).
    private func gemm(_ a: [Float], _ b: [Float], _ M: Int, _ K: Int, _ N: Int) throws -> [Float] {
        guard M > 0, K > 0, N > 0, a.count >= M * K, b.count >= K * N else {
            throw MelError.badResource
        }
        var c = [Float](repeating: 0, count: M * N)
        try a.withUnsafeBufferPointer { ap in
            try b.withUnsafeBufferPointer { bp in
                guard let aBaseAddress = ap.baseAddress,
                      let bBaseAddress = bp.baseAddress else {
                    throw MelError.badResource
                }
                vDSP_mmul(aBaseAddress, 1, bBaseAddress, 1, &c, 1,
                          vDSP_Length(M), vDSP_Length(N), vDSP_Length(K))
            }
        }
        return c
    }

    /// 16 kHz mono audio -> packed mel (nMels, nFrames) row-major. Uses the configured `nFrames`.
    /// Throws `MelError.audioTooShort` for clips shorter than the FFT half-window (instead of crashing).
    public func packedMel(_ audio: [Float]) throws -> [Float] {
        try packedMel(audio, frames: nFrames)
    }

    /// As above but for an explicit frame count (duration bucket) — the cos/sin/mel matrices are
    /// frame-independent, so one frontend serves every bucket. Audio shorter than `frames*hop` is
    /// zero-padded at the end (silent trailing frames), matching WhisperFeatureExtractor max-length.
    public func packedMel(_ audio: [Float], frames: Int) throws -> [Float] {
        let p = nFFT / 2
        guard audio.count > p else { throw MelError.audioTooShort(audio.count) }
        let needed = p + frames * hop + nFFT   // last frame base = (frames-1)*hop, +nFFT samples
        // center reflect-pad (numpy 'reflect': padded[j]=audio[p-j]) + the prefix this frame
        // window can read + trailing zeros. Longer tails must not affect capped model windows.
        var padded = [Float](repeating: 0, count: needed)
        for j in 0..<p { padded[j] = audio[Swift.min(p - j, audio.count - 1)] }
        let copiedSamples = Swift.min(audio.count, Swift.max(0, needed - p))
        for i in 0..<copiedSamples { padded[p + i] = audio[i] }

        // windowed frames F (frames, nFFT)
        var F = [Float](repeating: 0, count: frames * nFFT)
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { F[t * nFFT + n] = padded[base + n] * window[n] }
        }

        let real = try gemm(F, cosMat, frames, nFFT, nFreq)   // (frames, nFreq)
        let imag = try gemm(F, sinMat, frames, nFFT, nFreq)
        var power = [Float](repeating: 0, count: frames * nFreq)
        for i in 0..<power.count { power[i] = real[i] * real[i] + imag[i] * imag[i] }

        let mel = try gemm(power, melFilters, frames, nFreq, nMels)  // (frames, nMels)
        var logmel = mel.map { Foundation.log10(Swift.max($0, 1e-10)) }
        let gmax = logmel.max() ?? 0
        for i in 0..<logmel.count { logmel[i] = (Swift.max(logmel[i], gmax - 8.0) + 4.0) / 4.0 }

        // transpose (frames, nMels) -> packed (nMels, frames)
        var packed = [Float](repeating: 0, count: nMels * frames)
        for t in 0..<frames {
            for m in 0..<nMels { packed[m * frames + t] = logmel[t * nMels + m] }
        }
        return packed
    }
}
