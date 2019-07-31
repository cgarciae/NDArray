
import class Foundation.DispatchGroup
import class Foundation.DispatchQueue
import class Foundation.ProcessInfo

public let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

@inlinable
public func elementWise<A, B, C>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    apply f: (A, B) -> C
) -> NDArray<C> {
    precondition(ndArrayA.data.count == ndArrayB.data.count)
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayC = [C](unsafeUninitializedCapacity: nElements) { arrayC, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.withUnsafeBufferPointer { arrayB in
                for i in 0 ..< nElements {
                    arrayC[i] = f(
                        arrayA[ndArrayA.realIndex(of: i)],
                        arrayB[ndArrayA.realIndex(of: i)]
                    )
                }
            }
        }
    }

    return NDArray(
        arrayC,
        shape: ndArrayA.shape
    )
}

@inlinable
public func elementWiseInParallel<A, B, C>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> C
) -> NDArray<C> {
    precondition(ndArrayA.shape == ndArrayB.shape)
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayC = [C](unsafeUninitializedCapacity: nElements) { arrayC, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.withUnsafeBufferPointer { arrayB in
                let group = DispatchGroup()

                for range in splitRanges(total: nElements, splits: workers) {
                    group.enter()

                    DISPATCH.async { [arrayC] in
                        for i in range {
                            arrayC[i] = f(
                                arrayA[ndArrayA.realIndex(of: i)],
                                arrayB[ndArrayB.realIndex(of: i)]
                            )
                        }
                        group.leave()
                    }
                }

                group.wait()
            }
        }
    }

    return NDArray(
        arrayC,
        shape: ndArrayA.shape
    )
}

extension NDArray {
    @inlinable
    public func elementWiseApply<B, C>(_ other: NDArray<B>, _ f: (Scalar, B) -> C) -> NDArray<C> {
        elementWise(self, other, apply: f)
    }

    @inlinable
    public func elementWiseApplyInParallel<B, C>(
        _ other: NDArray<B>,
        workers: Int = CPU_COUNT,
        _ f: @escaping (Scalar, B) -> C
    ) -> NDArray<C> {
        elementWiseInParallel(self, other, workers: workers, apply: f)
    }
}