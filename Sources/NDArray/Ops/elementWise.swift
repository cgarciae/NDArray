
import Foundation

public let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

//////////////////////////////////////////////////////////////////////////////////////////
// 1
//////////////////////////////////////////////////////////////////////////////////////////

@inlinable
public func elementwise<A, Z>(
    _ ndArrayA: NDArray<A>,
    apply f: (A) -> Z
) -> NDArray<Z> {
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayZ = [Z](unsafeUninitializedCapacity: nElements) { arrayZ, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            for i in 0 ..< nElements {
                arrayZ[i] = f(
                    arrayA[ndArrayA.realIndex(of: i)]
                )
            }
        }
    }

    return NDArray(
        arrayZ,
        shape: ndArrayA.shape
    )
}

@inlinable
public func elementwiseInParallel<A, Z>(
    _ ndArrayA: NDArray<A>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A) -> Z
) -> NDArray<Z> {
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayZ = [Z](unsafeUninitializedCapacity: nElements) { arrayZ, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            let group = DispatchGroup()

            for range in splitRanges(total: nElements, splits: workers) {
                group.enter()

                DISPATCH.async { [arrayZ] in
                    for i in range {
                        arrayZ[i] = f(
                            arrayA[ndArrayA.realIndex(of: i)]
                        )
                    }
                    group.leave()
                }
            }

            group.wait()
        }
    }

    return NDArray(
        arrayZ,
        shape: ndArrayA.shape
    )
}

//////////////////////////////////////////////////////////////////////////////////////////
// 2
//////////////////////////////////////////////////////////////////////////////////////////

@inlinable
public func elementwise<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    apply f: (A, B) -> Z
) -> NDArray<Z> {
    precondition(ndArrayA.data.count == ndArrayB.data.count)
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayZ = [Z](unsafeUninitializedCapacity: nElements) { arrayZ, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.withUnsafeBufferPointer { arrayB in
                for i in 0 ..< nElements {
                    arrayZ[i] = f(
                        arrayA[ndArrayA.realIndex(of: i)],
                        arrayB[ndArrayB.realIndex(of: i)]
                    )
                }
            }
        }
    }

    return NDArray(
        arrayZ,
        shape: ndArrayA.shape
    )
}

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) -> NDArray<Z> {
    precondition(ndArrayA.shape == ndArrayB.shape)
    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayZ = [Z](unsafeUninitializedCapacity: nElements) { arrayZ, count in
        count = nElements

        ndArrayA.data.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.withUnsafeBufferPointer { arrayB in
                let group = DispatchGroup()

                for range in splitRanges(total: nElements, splits: workers) {
                    group.enter()

                    DISPATCH.async { [arrayZ] in
                        for i in range {
                            arrayZ[i] = f(
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
        arrayZ,
        shape: ndArrayA.shape
    )
}