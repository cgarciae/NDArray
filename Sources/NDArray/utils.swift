import class Foundation.DispatchGroup
import class Foundation.DispatchQueue
import class Foundation.ProcessInfo

let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

public func splitRanges(total: Int, splits: Int) -> [Range<Int>] {
    let total = Float(total)
    let points = Set(
        (0 ... splits).map { (s: Int) -> Int in
            let percent = Float(s) / Float(splits)
            return Int(total * percent)
        }
    )

    let orderedPoints = Array(points).sorted()
    var ranges = [Range<Int>]()

    for i in 0 ..< orderedPoints.count - 1 {
        ranges.append(orderedPoints[i] ..< orderedPoints[i + 1])
    }

    return ranges
}

public func elementWise<A, B, C>(
    _ arrayA: [A],
    _ arrayB: [B],
    apply f: (A, B) -> C
) -> [C] {
    precondition(arrayA.count == arrayB.count)

    let arrayC = [C](unsafeUninitializedCapacity: arrayA.count) { arrayC, count in
        count = arrayA.count

        arrayA.withUnsafeBufferPointer { arrayA in
            arrayB.withUnsafeBufferPointer { arrayB in
                for i in 0 ..< arrayA.count {
                    arrayC[i] = f(arrayA[i], arrayB[i])
                }
            }
        }
    }

    return arrayC
}

public func elementWiseInParallel<A, B, C>(
    _ arrayA: [A],
    _ arrayB: [B],
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> C
) -> [C] {
    precondition(arrayA.count == arrayB.count)

    let arrayC = [C](unsafeUninitializedCapacity: arrayA.count) { arrayC, count in
        count = arrayA.count

        arrayA.withUnsafeBufferPointer { arrayA in
            arrayB.withUnsafeBufferPointer { arrayB in
                let group = DispatchGroup()

                for range in splitRanges(total: count, splits: workers) {
                    group.enter()

                    DISPATCH.async { [arrayC] in
                        for i in range {
                            arrayC[i] = f(arrayA[i], arrayB[i])
                        }
                        group.leave()
                    }
                }

                group.wait()
            }
        }
    }

    return arrayC
}

extension NDArray {
    public func elementWiseApply<B, C>(_ other: NDArray<B>, _ f: (Scalar, B) -> C) -> NDArray<C> {
        NDArray<C>(
            elementWise(data, other.data, apply: f),
            shape: shape
        )
    }

    public func elementWiseApplyInParallel<B, C>(
        _ other: NDArray<B>,
        workers: Int = CPU_COUNT,
        _ f: @escaping (Scalar, B) -> C
    ) -> NDArray<C> {
        NDArray<C>(
            elementWiseInParallel(data, other.data, workers: workers, apply: f),
            shape: shape
        )
    }
}
