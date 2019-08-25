
import Foundation

public let DISPATCH = DispatchQueue(label: "NDArray", attributes: .concurrent)
public let CPU_COUNT = ProcessInfo.processInfo.activeProcessorCount

//////////////////////////////////////////////////////////////////////////////////////////
// 1
//////////////////////////////////////////////////////////////////////////////////////////

// @inlinable public func getIndexer<Scalar>(_ ndarray: NDArray<Scalar>) -> ((Int, UnsafeMutableBufferPointer<Int>)) -> Int {
//     ndarray.arrayShape.isOriginalShape ?
//         { $0.0 } : { ndarray.arrayShape.linearIndex(of: $0.1) }
// }

@inlinable
public func elementwise<A, Z>(
    _ ndArrayA: NDArray<A>,
    into ndArrayZ: inout NDArray<Z>,
    apply f: (A) -> Z
) {
    let nElements = ndArrayA.shape.product()

    ndArrayZ.withScalarSetter { zSetter in
        ndArrayA.withScalarGetter { aGetter in

            for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                let value = f(aGetter(index))
                zSetter(index, value)
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

    ndArrayZ.withScalarSetter { zSetter in
        ndArrayA.withScalarGetter { aGetter in

            let rangeMap = { indexSequence(range: $0, shape: ndArrayA.shape) }

            parFor(0 ..< nElements, rangeMap: rangeMap) { index in
                let value = f(aGetter(index))
                zSetter(index, value)
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

    ndArrayZ.withScalarSetter { zSetter in
        ndArrayA.withScalarGetter { aGetter in
            ndArrayB.withScalarGetter { bGetter in

                for index in indexSequence(range: 0 ..< nElements, shape: ndArrayA.shape) {
                    let value = f(aGetter(index), bGetter(index))
                    zSetter(index, value)
                }
            }
        }
    }
}

@inlinable
public func elementwiseAssignApply<A, B>(
    _ ndArrayA: inout NDArray<A>,
    _ ndArrayB: NDArray<B>,
    apply f: (A, B) -> A
) {
    var ndArrayB = ndArrayB

    if ndArrayA.shape != ndArrayB.shape {
        (ndArrayA, ndArrayB) = broadcast(ndArrayA, and: ndArrayB)
    }

    let nElements = ndArrayA.shape.product()

    let ndArrayAShape = ndArrayA.shape

    ndArrayA.withScalarGetterSetter { aGetterSetter in
        ndArrayB.withScalarGetter { bGetter in

            for index in indexSequence(range: 0 ..< nElements, shape: ndArrayAShape) {
                aGetterSetter(index) { aValue in
                    let value = f(aValue, bGetter(index))
                    return value
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

    if !ndArrayZ.anyNDArray.isSetable() {
        ndArrayZ = NDArray(ndArrayZ.baseCopy())
    }

    ndArrayZ.withScalarSetter { zSetter in
        ndArrayA.withScalarGetter { aGetter in
            ndArrayB.withScalarGetter { bGetter in

                let rangeMap = { indexSequence(range: $0, shape: ndArrayA.shape) }

                parFor(0 ..< nElements, rangeMap: rangeMap) { index in

                    let value = f(aGetter(index), bGetter(index))
                    zSetter(index, value)
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