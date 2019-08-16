
import Foundation

public let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

//////////////////////////////////////////////////////////////////////////////////////////
// 1
//////////////////////////////////////////////////////////////////////////////////////////

@inlinable public func getIndexer<Scalar>(_ ndarray: NDArray<Scalar>) -> ((Int, UnsafeMutableBufferPointer<Int>)) -> Int {
    ndarray.arrayShape.isOriginalShape ?
        { $0.0 } : { ndarray.arrayShape.linearIndex(of: $0.1) }
}

@inlinable
public func elementwise<A, Z>(
    _ ndArrayA: NDArray<A>,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A) -> Z
) {
    let nElements = ndArrayA.shape.product()

    ndArrayZ.data.value.withUnsafeMutableBufferPointer { arrayZ in
        ndArrayA.data.value.withUnsafeBufferPointer { arrayA in
            let allOriginalShape = ndArrayA.arrayShape.isOriginalShape &&
                ndArrayZ.arrayShape.isOriginalShape

            if allOriginalShape {
                for i in 0 ..< nElements {
                    arrayZ[i] = f(arrayA[i])
                }
            } else {
                let indexerA = getIndexer(ndArrayA)
                let indexerZ = getIndexer(ndArrayZ)

                // let indexerZ: ((Int, UnsafeMutableBufferPointer<Int>)) -> Int = ndArrayZ.arrayShape.isOriginalShape ?
                //     { $0.0 } : { ndArrayZ.arrayShape.linearIndex(of: $0.1) }

                for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                    let aIndex = indexerA(index)
                    let zIndex = indexerZ(index)

                    arrayZ[zIndex] = f(arrayA[aIndex])
                }
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
            let allOriginalShape = ndArrayA.arrayShape.isOriginalShape &&
                ndArrayZ.arrayShape.isOriginalShape

            if allOriginalShape {
                parFor(0 ..< nElements) { [arrayZ] i in
                    arrayZ[i] = f(arrayA[i])
                }
            } else {
                let indexerA = getIndexer(ndArrayA)
                let indexerZ = getIndexer(ndArrayZ)
                let rangeMap = { indexSequence(range: $0, shape: ndArrayA.shape) }

                parFor(0 ..< nElements, rangeMap: rangeMap) { [arrayZ] index in
                    let aIndex = indexerA(index)
                    let zIndex = indexerZ(index)

                    arrayZ[zIndex] = f(arrayA[aIndex])
                }
            }
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
                    let indexerA = getIndexer(ndArrayA)
                    let indexerB = getIndexer(ndArrayB)
                    let indexerZ = getIndexer(ndArrayZ)

                    for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                        let aIndex = indexerA(index)
                        let bIndex = indexerB(index)
                        let zIndex = indexerZ(index)

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
                let allOriginalShape = ndArrayA.arrayShape.isOriginalShape &&
                    ndArrayB.arrayShape.isOriginalShape &&
                    ndArrayZ.arrayShape.isOriginalShape

                if allOriginalShape {
                    parFor(0 ..< nElements) { [arrayZ] i in
                        arrayZ[i] = f(arrayA[i], arrayB[i])
                    }
                } else {
                    let rangeMap = { indexSequence(range: $0, shape: ndArrayA.shape) }
                    let indexerA = getIndexer(ndArrayA)
                    let indexerB = getIndexer(ndArrayB)
                    let indexerZ = getIndexer(ndArrayZ)

                    parFor(0 ..< nElements, rangeMap: rangeMap) { [arrayZ] index in
                        let aIndex = indexerA(index)
                        let bIndex = indexerB(index)
                        let zIndex = indexerZ(index)

                        arrayZ[zIndex] = f(arrayA[aIndex], arrayB[bIndex])
                    }
                }
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