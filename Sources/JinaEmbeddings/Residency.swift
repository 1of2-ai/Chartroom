import CoreML
import Foundation

/// Per-op compute-device distribution for a Core ML model — the authoritative ANE residency
/// measurement (`MLComputePlan`). `const` ops carry no device usage and are skipped.
public struct ResidencyReport: Sendable, CustomStringConvertible {
    public var ane = 0, cpu = 0, gpu = 0, other = 0
    public var total: Int { ane + cpu + gpu + other }
    public var anePercent: Double { total > 0 ? 100.0 * Double(ane) / Double(total) : 0 }

    public var description: String {
        String(format: "ops=%d  ANE=%d (%.1f%%)  CPU=%d  GPU=%d  other=%d",
               total, ane, anePercent, cpu, gpu, other)
    }
}

@available(macOS 14.4, iOS 17.4, *)
public func measureResidency(
    modelURL: URL,
    computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
    functionName: String? = nil
) async throws -> ResidencyReport {
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    if let functionName { config.functionName = functionName }
    let compiled = modelURL.pathExtension == "mlmodelc"
        ? modelURL : try await MLModel.compileModel(at: modelURL)
    let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: config)

    guard case let .program(program) = plan.modelStructure,
          let main = program.functions[functionName ?? "main"] else {
        throw NSError(domain: "Residency", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "function not found / not an mlprogram"])
    }

    var report = ResidencyReport()
    func walk(_ block: MLModelStructure.Program.Block) {
        for op in block.operations {
            if let usage = plan.deviceUsage(for: op) {
                switch usage.preferred {
                case .neuralEngine: report.ane += 1
                case .cpu: report.cpu += 1
                case .gpu: report.gpu += 1
                @unknown default: report.other += 1
                }
            }
            for nested in op.blocks { walk(nested) }
        }
    }
    walk(main.block)
    return report
}
