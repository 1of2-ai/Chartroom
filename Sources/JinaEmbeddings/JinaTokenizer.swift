import Foundation
import Tokenizers

/// Qwen2 byte-level BPE tokenizer (via swift-transformers), loaded offline from a local model
/// folder containing `tokenizer.json` + `tokenizer_config.json`. Token ids must match the HF
/// tokenizer exactly (verified by jina-tok against reference token_ids).
public struct JinaTokenizer {
    public let tokenizer: any Tokenizer

    public init(modelFolder: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    /// Encode raw text (already including any "Query: "/"Document: " prefix) to token ids.
    public func encode(_ text: String, addSpecialTokens: Bool = true) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens).map { Int32($0) }
    }
}
