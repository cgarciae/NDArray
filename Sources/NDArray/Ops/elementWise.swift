
import Foundation

public let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

//////////////////////////////////////////////////////////////////////////////////////////
// 1
//////////////////////////////////////////////////////////////////////////////////////////

@inlinable
public func elementwise<A, Z>(
    _ ndArrayA: NDArray<A>,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A) -> Z
) {
    let nElements = ndArrayA.shape.product()

    ndArrayZ.data.value.withUnsafeMutableBufferPointer { arrayZ in
        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            for (_, rectangularIndex) in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                let aIndex = ndArrayA.arrayShape.linearIndex(of: rectangularIndex.value)
                let zIndex = ndArrayZ.arrayShape.linearIndex(of: rectangularIndex.value)

                arrayZ[zIndex] = f(arrayA[aIndex])
            }
        }
    }
}

@inlinable
public func elementwise<A, Z>(
    _ ndArrayA: NDArray<A>,
    apply f: (A) -> Z
) -> NDArray<Z> {
    let nElements = ndArrayA.shape.product()

    var ndArrayZ = NDArray<Z>(
        [Z](unsafeUninitializedCapacity: nElements) { x, count in
            count = nElements
        },
        shape: ndArrayA.shape
    )

    elementwise(ndArrayA, into: &ndArrayZ, apply: f)

    return ndArrayZ
}

@inlinable
public func elementwiseInParallel<A, Z>(
    _ ndArrayA: NDArray<A>,
    into ndArrayZ: inout NDArray<Z>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A) -> Z
) {
    let nElements = ndArrayA.shape.product()

    ndArrayZ.data.value.withUnsafeMutableBufferPointer { arrayZ in
        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            let group = DispatchGroup()

            for range in splitRanges(total: nElements, splits: workers) {
                group.enter()

                DISPATCH.async { [arrayZ, ndArrayZ] in

                    for (_, rectangularIndex) in indexSequence(range: range, shape: ndArrayA.shape) {
                        let aIndex = ndArrayA.arrayShape.linearIndex(of: rectangularIndex.value)
                        let zIndex = ndArrayZ.arrayShape.linearIndex(of: rectangularIndex.value)

                        arrayZ[zIndex] = f(arrayA[aIndex])
                    }

                    group.leave()
                }
            }

            group.wait()
        }
    }
}

@inlinable
public func elementwiseInParallel<A, Z>(
    _ ndArrayA: NDArray<A>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A) -> Z
) -> NDArray<Z> {
    let nElements = ndArrayA.shape.product()

    var ndArrayZ = NDArray<Z>(
        [Z](unsafeUninitializedCapacity: nElements) { x, count in
            count = nElements
        },
        shape: ndArrayA.shape
    )

    elementwiseInParallel(ndArrayA, into: &ndArrayZ, workers: workers, apply: f)

    return ndArrayZ
}

//////////////////////////////////////////////////////////////////////////////////////////
// 2
//////////////////////////////////////////////////////////////////////////////////////////

@inlinable
public func elementwise<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A, B) -> Z
) {
    var ndArrayA = ndArrayA
    var ndArrayB = ndArrayB

    if ndArrayA.shape != ndArrayB.shape {
        (ndArrayA, ndArrayB) = broadcast(ndArrayA, and: ndArrayB)
    }

    let nElements = ndArrayA.shape.product()

    ndArrayZ.data.value.withUnsafeMutableBufferPointer { arrayZ in
        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.value.withUnsafeBufferPointer { arrayB in
                let allOriginalShape = ndArrayA.arrayShape.isOriginalShape &&
                    ndArrayB.arrayShape.isOriginalShape &&
                    ndArrayZ.arrayShape.isOriginalShape

                if allOriginalShape {
                    for i in 0 ..< nElements {
                        arrayZ[i] = f(arrayA[i], arrayB[i])
                    }
                } else {
                    for (_, rectangularIndex) in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                        let aIndex = ndArrayA.arrayShape.linearIndex(of: rectangularIndex.value)
                        let bIndex = ndArrayB.arrayShape.linearIndex(of: rectangularIndex.value)
                        let zIndex = ndArrayZ.arrayShape.linearIndex(of: rectangularIndex.value)

                        arrayZ[zIndex] = f(arrayA[aIndex], arrayB[bIndex])
                    }
                }
            }
        }
    }
}

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

    let nElements = ndArrayA.shape.product()

    var ndArrayZ = NDArray<Z>(
        [Z](unsafeUninitializedCapacity: nElements) { x, count in
            count = nElements
        },
        shape: ndArrayA.shape
    )

    elementwise(ndArrayA, ndArrayB, into: &ndArrayZ, apply: f)

    return ndArrayZ
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
    _ ndArrayA: NDArray<A>,
    _ b: B,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A, B) -> Z
) {
    elementwise(ndArrayA, into: &ndArrayZ) { a in f(a, b) }
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
public func elementwise<A, B, Z>(
    _ a: A,
    _ ndArrayB: NDArray<B>,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A, B) -> Z
) {
    elementwise(ndArrayB, into: &ndArrayZ) { b in f(a, b) }
}

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    into ndArrayZ: inout NDArray<Z>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) {
    precondition(ndArrayA.shape == ndArrayB.shape)
    let nElements = ndArrayA.shape.product()

    ndArrayZ.data.value.withUnsafeMutableBufferPointer { arrayZ in
        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            ndArrayB.data.value.withUnsafeBufferPointer { arrayB in
                let group = DispatchGroup()

                for range in splitRanges(total: nElements, splits: workers) {
                    group.enter()

                    DISPATCH.async { [ndArrayZ, arrayZ] in
                        for (_, rectangularIndex) in indexSequence(range: range, shape: ndArrayA.shape) {
                            let aIndex = ndArrayA.arrayShape.linearIndex(of: rectangularIndex.value)
                            let bIndex = ndArrayB.arrayShape.linearIndex(of: rectangularIndex.value)
                            let zIndex = ndArrayZ.arrayShape.linearIndex(of: rectangularIndex.value)

                            arrayZ[zIndex] = f(arrayA[aIndex], arrayB[bIndex])
                        }
                        group.leave()
                    }
                }

                group.wait()
            }
        }
    }
}

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ ndArrayA: NDArray<A>,
    _ ndArrayB: NDArray<B>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) -> NDArray<Z> {
    var ndArrayA = ndArrayA
    var ndArrayB = ndArrayB

    if ndArrayA.shape != ndArrayB.shape {
        (ndArrayA, ndArrayB) = broadcast(ndArrayA, and: ndArrayB)
    }

    let nElements = ndArrayA.shape.product()

    var ndArrayZ = NDArray<Z>(
        [Z](unsafeUninitializedCapacity: nElements) { x, count in
            count = nElements
        },
        shape: ndArrayA.shape
    )

    elementwiseInParallel(ndArrayA, ndArrayB, into: &ndArrayZ, workers: workers, apply: f)

    return ndArrayZ
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
    _ ndArrayA: NDArray<A>,
    _ b: B,
    into ndArrayZ: inout NDArray<Z>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) {
    elementwiseInParallel(ndArrayA, into: &ndArrayZ, workers: workers) { a in f(a, b) }
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

@inlinable
public func elementwiseInParallel<A, B, Z>(
    _ a: A,
    _ ndArrayB: NDArray<B>,
    into ndArrayZ: inout NDArray<Z>,
    workers: Int = CPU_COUNT,
    apply f: @escaping (A, B) -> Z
) {
    elementwiseInParallel(ndArrayB, into: &ndArrayZ, workers: workers) { b in f(a, b) }
}