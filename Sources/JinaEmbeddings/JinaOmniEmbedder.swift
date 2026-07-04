import CoreML
import Foundation

/// One entry point for the whole model. Point it at a directory of Core ML artifacts and call
/// `embed(...)` for any modality — text, image, audio, or video — all landing in the same
/// L2-normalized 1024-d space (compare any two with ``cosine(_:_:)``).
///
/// ```swift
/// import JinaEmbeddings
///
/// let jina = JinaOmniEmbedder(modelsDirectory: url)   // instant — nothing is loaded yet
///
/// let q   = try await jina.embed(text: "What is the capital of France?", prompt: .query)
/// let d   = try await jina.embed(text: "Paris is the capital of France.", prompt: .document)
/// let img = try jina.embed(imageURL: photoURL)        // any resolution
/// let aud = try jina.embed(audioURL: clipURL)         // any length up to ~30 s
/// let vid = try await jina.embed(videoURL: movieURL)  // .mp4 → evenly-spaced frames
///
/// let similarity = cosine(q, d)                        // cross-modal too: cosine(q, img)
/// let small = try await jina.embed(text: "search", prompt: .query, dim: 256)   // Matryoshka
/// ```
///
/// **Lazy & lean.** Each modality's Core ML towers are compiled and loaded on first use and then
/// cached, so an app that only embeds text never pays to load the vision/audio towers. Construction
/// is thread-safe; once built, a single embedder is safe to call concurrently (the underlying
/// embedders guard their own caches).
///
/// **Why text and video file embedding are `async`.** The text path's tokenizer loads asynchronously
/// (swift-transformers), and video frame extraction uses AVFoundation's current async image-generator
/// APIs. Image, audio, and raw video-frame embedding remain synchronous; call them from your own
/// background context for heavy batches, or reach for the underlying embedders for full control.
///
/// **Artifacts are external.** The `.mlpackage`s are gigabytes and are not shipped with this package.
/// Conversion produces a ``JinaModelBundle``; inference reads the bundle manifest and lazy-loads the
/// artifacts it names.
public final class JinaOmniEmbedder: @unchecked Sendable {

    /// A modality this embedder can produce vectors for. Used in ``JinaOmniError`` messages.
    public enum Modality: String, Sendable, CaseIterable {
        case text, image, audio, video
    }

    /// Filenames (resolved against the model bundle) and compute placement for each tower.
    ///
    /// Defaults match the v1 ``JinaModelBundle`` manifest. Any field may be an absolute path
    /// (starting with `/`), in which case the bundle directory is ignored for that file.
    public struct Configuration: Sendable {
        // MARK: Core ML packages
        /// Multi-function text tower (`bucket_<S>` functions). Default `text_multifunc.mlpackage`.
        public var textModel: String
        /// Runtime-position masked ViT for images. Default `vision_tower_masked_multifunc.mlpackage`.
        public var imageModel: String
        /// Per-frame block-diagonal ViT for video. Default `vision_tower_video_multifunc.mlpackage`.
        public var videoModel: String
        /// Runtime-masked audio encoder. Default `audio_tower_masked_multifunc.mlpackage`.
        public var audioModel: String
        /// Shared general decoder, part 1 (input_ids → inputs_embeds). Default `embed_multifunc.mlpackage`.
        public var embedModel: String
        /// Shared general decoder, part 2 (inputs_embeds → embedding). Default `decoder_embeds_multifunc.mlpackage`.
        public var decoderModel: String

        // MARK: Host-side resources
        /// Tokenizer folder (`tokenizer.json` + `tokenizer_config.json`). Default `jina-v5-omni-small`.
        public var tokenizerFolder: String
        /// Vision resources folder holding `meta.json`, `pos_embed_table.f32`, `rope_inv_freq.f32`
        /// (produced by `python/parity/export_vision_swift_refs.py`). Default `vision_swift`.
        public var visionResourcesFolder: String

        // MARK: Knobs
        /// Sequence-length buckets compiled into the text tower. Default `[32, 64, 128, 256, 512]`.
        public var textBuckets: [Int]
        /// Text tower placement. `nil` (default) = adaptive (ANE for short buckets, GPU for long).
        public var textComputeUnits: MLComputeUnits?
        /// Image/audio/video *encoder* placement. `.cpuAndGPU` (default) is most accurate AND fastest.
        public var encoderUnits: MLComputeUnits
        /// General decoder placement. `nil` (default) = adaptive (ANE for S≤256, GPU for S≥512).
        public var decoderUnits: MLComputeUnits?

        public init(
            textModel: String = "text_multifunc.mlpackage",
            imageModel: String = "vision_tower_masked_multifunc.mlpackage",
            videoModel: String = "vision_tower_video_multifunc.mlpackage",
            audioModel: String = "audio_tower_masked_multifunc.mlpackage",
            embedModel: String = "embed_multifunc.mlpackage",
            decoderModel: String = "decoder_embeds_multifunc.mlpackage",
            tokenizerFolder: String = "jina-v5-omni-small",
            visionResourcesFolder: String = "vision_swift",
            textBuckets: [Int] = [32, 64, 128, 256, 512],
            textComputeUnits: MLComputeUnits? = nil,
            encoderUnits: MLComputeUnits = .cpuAndGPU,
            decoderUnits: MLComputeUnits? = nil
        ) {
            self.textModel = textModel
            self.imageModel = imageModel
            self.videoModel = videoModel
            self.audioModel = audioModel
            self.embedModel = embedModel
            self.decoderModel = decoderModel
            self.tokenizerFolder = tokenizerFolder
            self.visionResourcesFolder = visionResourcesFolder
            self.textBuckets = textBuckets
            self.textComputeUnits = textComputeUnits
            self.encoderUnits = encoderUnits
            self.decoderUnits = decoderUnits
        }

        /// Build a configuration from the artifact paths declared by a model bundle manifest, keeping
        /// compute placement as an inference-side decision.
        public init(
            manifest: JinaModelBundle.Manifest,
            textComputeUnits: MLComputeUnits? = nil,
            encoderUnits: MLComputeUnits = .cpuAndGPU,
            decoderUnits: MLComputeUnits? = nil
        ) {
            self.init(
                textModel: manifest.text.model,
                imageModel: manifest.image.encoder,
                videoModel: manifest.video.encoder,
                audioModel: manifest.audio.encoder,
                embedModel: manifest.decoder.embed,
                decoderModel: manifest.decoder.model,
                tokenizerFolder: manifest.text.tokenizer,
                visionResourcesFolder: manifest.image.resources,
                textBuckets: manifest.text.buckets,
                textComputeUnits: textComputeUnits,
                encoderUnits: encoderUnits,
                decoderUnits: decoderUnits
            )
        }

        /// The defaults (GPU encoders + adaptive decoder) — recommended for accuracy and speed.
        public static let `default` = Configuration()

        /// Lowest-power, GPU-free deployment: run *every* tower on the Neural Engine. Slower than the
        /// hybrid default and slightly less accurate, but still at/above the model's native bf16 floor.
        public static func fullANE(_ base: Configuration = Configuration()) -> Configuration {
            var c = base
            c.textComputeUnits = .cpuAndNeuralEngine
            c.encoderUnits = .cpuAndNeuralEngine
            c.decoderUnits = .cpuAndNeuralEngine
            return c
        }

        /// Full-ANE deployment for a bundle manifest, preserving the artifact paths declared by
        /// conversion while changing only inference-side compute placement.
        public static func fullANE(manifest: JinaModelBundle.Manifest) -> Configuration {
            fullANE(Configuration(manifest: manifest))
        }
    }

    /// Convenience alias so callers can write `.query` without naming `JinaTextEmbedder`.
    public typealias Prompt = JinaTextEmbedder.Prompt

    public let modelBundle: JinaModelBundle
    public var modelsDirectory: URL { modelBundle.rootDirectory }
    public var modelID: String { modelBundle.manifest.modelID }
    public var embeddingDimension: Int { modelBundle.manifest.embeddingDimension }
    public let configuration: Configuration

    private let lock = NSLock()                              // guards the lazy caches below
    private var _text: JinaTextEmbedder?
    private var _image: JinaImageEmbedderMasked?
    private var _audio: JinaAudioEmbedderMasked?
    private var _video: JinaVideoEmbedderMasked?

    /// - Parameters:
    ///   - modelsDirectory: Directory containing the Core ML packages, tokenizer folder, and vision
    ///     resources (see ``Configuration`` for the expected names).
    ///   - configuration: Filenames and compute-unit placement. Defaults to ``Configuration/default``.
    ///
    /// Construction is instant — no files are touched until the first `embed(...)` for that modality.
    public init(modelsDirectory: URL, configuration: Configuration = .default) {
        self.modelBundle = .legacy(directory: modelsDirectory)
        self.configuration = configuration
    }

    /// Preferred initializer for converted model bundles that include `manifest.json`.
    public init(bundle: JinaModelBundle) {
        self.modelBundle = bundle
        self.configuration = Configuration(manifest: bundle.manifest)
    }

    /// Initialize from a bundle, overriding artifact names or compute placement when needed.
    public init(bundle: JinaModelBundle, configuration: Configuration) {
        self.modelBundle = bundle
        self.configuration = configuration
    }

    // MARK: - Embedding

    /// Embed text. `prompt` adds the model's task prefix (`.query` / `.document`); `dim` truncates to
    /// a Matryoshka dimension (re-normalized). Returns an L2-normalized `[Float]`.
    public func embed(text: String, prompt: Prompt = .none, dim: Int? = nil) async throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try await textEmbedder().embed(text, prompt: prompt, dim: dim), dim: dim)
    }

    /// Batch text embedding: groups by bucket internally and returns vectors in input order.
    public func embed(batch texts: [String], prompt: Prompt = .none, dim: Int? = nil) async throws -> [[Float]] {
        try validateRequestedDimension(dim)
        let embeddings = try await textEmbedder().embed(batch: texts, prompt: prompt, dim: dim)
        return try embeddings.map { try validateEmbedding($0, dim: dim) }
    }

    /// Embed an image file of any resolution (smart-resized to the model grid).
    public func embed(imageURL: URL, dim: Int? = nil) throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try imageEmbedder().embed(imageURL: imageURL, dim: dim), dim: dim)
    }

    /// Embed a raw row-major RGB buffer (`h*w*3`), with `h`/`w` aligned to the model's 32-px factor.
    /// Exact (no resample) — the full host path.
    public func embed(imageRGB rgb: [UInt8], h: Int, w: Int, dim: Int? = nil) throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try imageEmbedder().embed(rgb: rgb, h: h, w: w, dim: dim), dim: dim)
    }

    /// Embed an audio file (decoded to 16 kHz mono via AVFoundation). Any length up to ~30 s.
    public func embed(audioURL: URL, dim: Int? = nil) throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try audioEmbedder().embed(audioURL: audioURL, dim: dim), dim: dim)
    }

    /// Embed a 16 kHz mono waveform. Any length up to ~30 s (longer is truncated to the model's limit).
    public func embed(audio waveform: [Float], dim: Int? = nil) throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try audioEmbedder().embed(waveform, dim: dim), dim: dim)
    }

    /// Embed a video file: extract `frameCount` evenly-spaced frames at `frameSize`², then run the ViT.
    /// `frameSize` must be a multiple of 32; `frameCount/2 · (frameSize/16)²` must fit a video bucket.
    public func embed(videoURL: URL, frameCount: Int = 8, frameSize: Int = 256, dim: Int? = nil) async throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(
            try await videoEmbedder().embed(videoURL: videoURL, frameCount: frameCount, frameSize: frameSize, dim: dim),
            dim: dim
        )
    }

    /// Embed raw RGB video frames (count = `2·t`, each `h*w*3`, factor-aligned) — reference-exact path.
    public func embed(videoFrames frames: [[UInt8]], h: Int, w: Int, dim: Int? = nil) throws -> [Float] {
        try validateRequestedDimension(dim)
        return try validateEmbedding(try videoEmbedder().embed(frames: frames, h: h, w: w, dim: dim), dim: dim)
    }

    // MARK: - Underlying embedders (lazily built, cached, thread-safe)

    /// The text embedder, built and cached on first use. Building is awaited (async tokenizer load);
    /// the cache reads/writes happen in synchronous helpers (`NSLock` is unavailable in async code).
    /// Internal: `JinaTextEmbedder` is not `Sendable`, so it can't be returned across an actor
    /// boundary — text is consumed via ``embed(text:prompt:dim:)``, which returns a `Sendable` `[Float]`.
    public func textEmbedder() async throws -> JinaTextEmbedder {
        if let t = cachedText() { return t }
        let built = try await Self.buildText(dir: modelsDirectory, config: configuration)
        return storeText(built)   // a concurrent first-caller may win; keep the single cached instance
    }

    private func cachedText() -> JinaTextEmbedder? {
        lock.lock(); defer { lock.unlock() }
        return _text
    }

    private func storeText(_ t: JinaTextEmbedder) -> JinaTextEmbedder {
        lock.lock(); defer { lock.unlock() }
        if let existing = _text { return existing }
        _text = t
        return t
    }

    /// The image embedder, built and cached on first call. Like text, the multi-second
    /// Core ML load happens outside the shared lock so it never stalls embeds that only
    /// need a cache read; a concurrent first-caller may win and the extra build is dropped.
    public func imageEmbedder() throws -> JinaImageEmbedderMasked {
        if let e = cachedImage() { return e }
        let vis = resolve(configuration.imageModel), emb = resolve(configuration.embedModel)
        let dec = resolve(configuration.decoderModel), res = resolve(configuration.visionResourcesFolder)
        try require(vis, .image); try require(emb, .image); try require(dec, .image); try require(res, .image)
        let built = try JinaImageEmbedderMasked(
            visionModelURL: vis, embedModelURL: emb, decoderModelURL: dec, resourcesDir: res,
            encoderUnits: configuration.encoderUnits, decoderUnits: configuration.decoderUnits)
        return storeImage(built)
    }

    private func cachedImage() -> JinaImageEmbedderMasked? {
        lock.lock(); defer { lock.unlock() }
        return _image
    }

    private func storeImage(_ e: JinaImageEmbedderMasked) -> JinaImageEmbedderMasked {
        lock.lock(); defer { lock.unlock() }
        if let existing = _image { return existing }
        _image = e
        return e
    }

    /// The audio embedder, built and cached on first call (build outside the lock; see image).
    public func audioEmbedder() throws -> JinaAudioEmbedderMasked {
        if let e = cachedAudio() { return e }
        let aud = resolve(configuration.audioModel), emb = resolve(configuration.embedModel)
        let dec = resolve(configuration.decoderModel)
        try require(aud, .audio); try require(emb, .audio); try require(dec, .audio)
        let built = try JinaAudioEmbedderMasked(
            audioModelURL: aud, embedModelURL: emb, decoderModelURL: dec,
            encoderUnits: configuration.encoderUnits, decoderUnits: configuration.decoderUnits)
        return storeAudio(built)
    }

    private func cachedAudio() -> JinaAudioEmbedderMasked? {
        lock.lock(); defer { lock.unlock() }
        return _audio
    }

    private func storeAudio(_ e: JinaAudioEmbedderMasked) -> JinaAudioEmbedderMasked {
        lock.lock(); defer { lock.unlock() }
        if let existing = _audio { return existing }
        _audio = e
        return e
    }

    /// The video embedder, built and cached on first call (build outside the lock; see image).
    public func videoEmbedder() throws -> JinaVideoEmbedderMasked {
        if let e = cachedVideo() { return e }
        let vis = resolve(configuration.videoModel), emb = resolve(configuration.embedModel)
        let dec = resolve(configuration.decoderModel), res = resolve(configuration.visionResourcesFolder)
        try require(vis, .video); try require(emb, .video); try require(dec, .video); try require(res, .video)
        let built = try JinaVideoEmbedderMasked(
            visionModelURL: vis, embedModelURL: emb, decoderModelURL: dec, resourcesDir: res,
            encoderUnits: configuration.encoderUnits, decoderUnits: configuration.decoderUnits)
        return storeVideo(built)
    }

    private func cachedVideo() -> JinaVideoEmbedderMasked? {
        lock.lock(); defer { lock.unlock() }
        return _video
    }

    private func storeVideo(_ e: JinaVideoEmbedderMasked) -> JinaVideoEmbedderMasked {
        lock.lock(); defer { lock.unlock() }
        if let existing = _video { return existing }
        _video = e
        return e
    }

    // MARK: - Helpers

    private static func buildText(dir: URL, config: Configuration) async throws -> JinaTextEmbedder {
        let model = resolve(dir: dir, config.textModel)
        let tokenizer = resolve(dir: dir, config.tokenizerFolder)
        try require(model, .text); try require(tokenizer, .text)
        return try await JinaTextEmbedder(
            multiFunctionModelURL: model, tokenizerFolder: tokenizer,
            buckets: config.textBuckets, computeUnits: config.textComputeUnits)
    }

    private func resolve(_ name: String) -> URL { modelBundle.resolve(name) }

    private static func resolve(dir: URL, _ name: String) -> URL {
        JinaModelBundle.resolve(rootDirectory: dir, path: name)
    }

    private func require(_ url: URL, _ modality: Modality) throws { try Self.require(url, modality) }

    private static func require(_ url: URL, _ modality: Modality) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            throw JinaOmniError.missingArtifact(modality: modality, url: url)
        }
    }

    private func validateRequestedDimension(_ dim: Int?) throws {
        guard let dim else { return }
        guard (1...embeddingDimension).contains(dim) else {
            throw JinaOmniError.invalidEmbeddingDimension(requested: dim, maximum: embeddingDimension)
        }
    }

    private func validateEmbedding(_ embedding: [Float], dim: Int?) throws -> [Float] {
        let expected = dim ?? embeddingDimension
        guard embedding.count == expected else {
            throw JinaOmniError.embeddingDimensionMismatch(expected: expected, actual: embedding.count)
        }
        return embedding
    }
}

/// Errors surfaced by ``JinaOmniEmbedder`` before Core ML is ever invoked.
public enum JinaOmniError: Error, CustomStringConvertible {
    /// A file/folder a modality needs was not found at the resolved path.
    case missingArtifact(modality: JinaOmniEmbedder.Modality, url: URL)
    case invalidEmbeddingDimension(requested: Int, maximum: Int)
    case embeddingDimensionMismatch(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .missingArtifact(modality, url):
            return "JinaOmniEmbedder: missing \(modality.rawValue) artifact at \(url.path). "
                + "Build it with python/build_all.sh and place it in the models directory "
                + "(or set the name in JinaOmniEmbedder.Configuration)."
        case let .invalidEmbeddingDimension(requested, maximum):
            return "JinaOmniEmbedder: invalid embedding dimension \(requested). Expected 1...\(maximum)."
        case let .embeddingDimensionMismatch(expected, actual):
            return "JinaOmniEmbedder: embedding dimension mismatch. Expected \(expected), got \(actual)."
        }
    }
}
