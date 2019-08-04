
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

        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            // for i in 0 ..< nElements {
            //     arrayZ[i] = f(
            //         arrayA[ndArrayA.realIndex(of: i)]
            //     )
            // }
            for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                arrayZ[index.linearIndex] = f(
                    arrayA[ndArrayA.arrayShape.linearIndex(of: index.rectangularIndex.value)]
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

        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            let group = DispatchGroup()

            for range in splitRanges(total: nElements, splits: workers) {
                group.enter()

                DISPATCH.async { [arrayZ] in

                    for index in indexSequence(range: range, shape: ndArrayA.shape) {
                        arrayZ[index.linearIndex] = f(
                            arrayA[ndArrayA.arrayShape.linearIndex(of: index.rectangularIndex.value)]
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
    var ndArrayA = ndArrayA
    var ndArrayB = ndArrayB

    if ndArrayA.shape != ndArrayB.shape {
        (ndArrayA, ndArrayB) = broadcast(ndArrayA, and: ndArrayB)
    }

    let nElements = ndArrayA.shape.reduce(1, *)

    let arrayZ = [Z](unsafeUninitializedCapacity: nElements) { arrayZ, count in
        count = nElements

        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.value.withUnsafeBufferPointer { arrayB in
                for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                    arrayZ[index.linearIndex] = f(
                        arrayA[ndArrayA.arrayShape.linearIndex(of: index.rectangularIndex.value)],
                        arrayB[ndArrayB.arrayShape.linearIndex(of: index.rectangularIndex.value)]
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
public func elementwise<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ b: B,
    apply f: (A, B) -> Z
) -> NDArray<Z> {
    elementwise(ndArrayA) { a in f(a, b) }
}

@inlinable
public func elementwise<A, B, Z>(
    _ a: A,
    _ ndArrayB: NDArray<B>,
    apply f: (A, B) -> Z
) -> NDArray<Z> {
    elementwise(ndArrayB) { b in f(a, b) }
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

        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.value.withUnsafeBufferPointer { arrayB in
                let group = DispatchGroup()

                for range in splitRanges(total: nElements, splits: workers) {
                    group.enter()

                    DISPATCH.async { [arrayZ] in
                        for index in indexSequence(range: range, shape: ndArrayA.shape) {
                            arrayZ[index.linearIndex] = f(
                                arrayA[ndArrayA.arrayShape.linearIndex(of: index.rectangularIndex.value)],
                                arrayB[ndArrayB.arrayShape.linearIndex(of: index.rectangularIndex.value)]
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

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ b: B,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) -> NDArray<Z> {
    elementwiseInParallel(ndArrayA, workers: workers) { a in f(a, b) }
}

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ a: A,
    _ ndArrayB: NDArray<B>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) -> NDArray<Z> {
    elementwiseInParallel(ndArrayB, workers: workers) { b in f(a, b) }
}