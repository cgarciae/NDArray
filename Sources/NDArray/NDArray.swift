public protocol MultiArray {
    associatedtype Element

    var array: [Element] { get }
}

extension Array : MultiArray {
    public var array: [Element] { self }
}

public protocol NDArrayProtocol {
    associatedtype Scalar

    typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    var shape: [Int] { get }

    func subscript_get(_: [ArrayRange]) -> NDArray<Scalar>
    mutating func subscript_set(_: [ArrayRange], _: NDArray<Scalar>) -> Void
    func linearIndex(at indexes: [Int]) -> Int
    func dataValue(at indexes: [Int]) -> Scalar
    func withScalarGetter(_: (@escaping ScalarGetter) -> Void)
    mutating func withScalarSetter(_: (@escaping ScalarSetter) -> Void)

    // ops
    func transposed(_: [Int]) -> NDArray<Scalar>
    func tiled(by: [Int]) -> NDArray<Scalar>
    func expandDimensions(axis: Int) -> NDArray<Scalar>
    func scalarized() -> Scalar

    //
    func toArray<T: MultiArray>(_: T.Type) -> T

    //
    // mutating func copyInternals() -> Void
    func copy() -> Self
}

extension NDArrayProtocol {
    public func toArray<T: MultiArray>(_: T.Type) -> T {
        let cp: BaseNDArray = getBase()
        var array: [Any] = cp.data.value

        for n in cp.shape.reversed().dropLast() {
            array = array.chunked(into: n) as [Any]
        }

        return array as! T
    }
}

public struct NDArray<Scalar>: NDArrayProtocol {
    public let shape: [Int]

    public typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    public typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    public var anyNDArray: AnyNDArray<Scalar>

    public init<N:NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        shape = ndarray.shape
        anyNDArray = AnyNDArray(ndarray)
    }

    internal init(_ anyNDArray: AnyNDArray<Scalar>) {
        shape = anyNDArray.shape
        self.anyNDArray = anyNDArray
    }

    init(_ ndarray: NDArray) {
        self = ndarray
    }

    public init(_ data: [Any], shape: [Int]? = nil) {
        let (flatData, calculatedShape): ([Scalar], [Int]) = flattenArrays(data)

        precondition(
            calculatedShape.product() == flatData.count,
            "All sub-arrays in data must have equal length. Calculated shape: \(calculatedShape), \(flatData)"
        )

        if let shape = shape {
            precondition(
                shape.product() == flatData.count,
                "Invalid shape, number of elements"
            )
        }

        let data = Ref(flatData)
        let arrayShape = ArrayShape(shape ?? calculatedShape)

        self = NDArray(BaseNDArray(data, shape: arrayShape))
    }

    public init(_ data: Scalar) {
        self = NDArray(BaseNDArray(
            Ref([data]),
            shape: ArrayShape(
                [DimensionProtocol](),
                linearMemoryOffset: 0
            )
        ))
    }

    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        anyNDArray._subscript_get(ranges)
    }

    public mutating func subscript_set(_ ranges: [ArrayRange], _ value: NDArray<Scalar>) {
        if !isKnownUniquelyReferenced(&anyNDArray) {
            anyNDArray = anyNDArray._copy()
        }
        anyNDArray._subscript_set(ranges, value)
    }

    public func linearIndex(at indexes: [Int]) -> Int {
        anyNDArray._linearIndex(indexes)
    }

    public func dataValue(at indexes: [Int]) -> Scalar {
        anyNDArray._dataValue(indexes)
    }

    public func withScalarGetter(_ body: (@escaping ScalarGetter) -> Void) {
        anyNDArray._withScalarGetter(body)
    }

    public mutating func withScalarSetter(_ body: (@escaping ScalarSetter) -> Void) {
        anyNDArray._withScalarSetter(body)
    }

    public func transposed(_ permutations: [Int]) -> NDArray<Scalar> {
        anyNDArray._transposed(permutations)
    }

    public func tiled(by amounts: [Int]) -> NDArray<Scalar> {
        anyNDArray._tiled(amounts)
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        anyNDArray._expandDimensions(axis)
    }

    public func scalarized() -> Scalar {
        anyNDArray._scalarized()
    }

    // public mutating func copyInternals() {
    //     anyNDArray._copyInternals()
    // }

    public func copy() -> NDArray<Scalar> {
        NDArray(anyNDArray._copy())
    }
}

public class AnyNDArray<Scalar> {
    public let shape: [Int]

    public typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    public typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    let _linearIndex: ([Int]) -> Int
    let _dataValue: ([Int]) -> Scalar
    let _subscript_get: ([ArrayRange]) -> NDArray<Scalar>
    let _subscript_set: ([ArrayRange], NDArray<Scalar>) -> Void
    let _withScalarGetter: ((@escaping ScalarGetter) -> Void) -> Void
    let _withScalarSetter: ((@escaping ScalarSetter) -> Void) -> Void
    let _transposed: ([Int]) -> NDArray<Scalar>
    let _tiled: ([Int]) -> NDArray<Scalar>
    let _expandDimensions: (Int) -> NDArray<Scalar>
    let _scalarized: () -> Scalar
    // let _copyInternals: () -> Void
    let _copy: () -> AnyNDArray<Scalar>

    public init<N: NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        var ndarray = ndarray

        shape = ndarray.shape
        _linearIndex = { ndarray.linearIndex(at: $0) }
        _dataValue = { ndarray.dataValue(at: $0) }
        _subscript_get = { ndarray.subscript_get($0) }
        _subscript_set = {
            ndarray.subscript_set($0, $1)
        }
        _withScalarGetter = { ndarray.withScalarGetter($0) }
        _withScalarSetter = { ndarray.withScalarSetter($0) }
        _transposed = { ndarray.transposed($0) }
        _tiled = { ndarray.tiled(by: $0) }
        _expandDimensions = { ndarray.expandDimensions(axis: $0) }
        _scalarized = { ndarray.scalarized() }
        // _copyInternals = { ndarray.copyInternals() }
        _copy = { AnyNDArray(ndarray.copy()) }
    }

    // init(_ ndarray: AnyNDArray) {
    //     self = ndarray
    // }

    // public convenience init(_ data: [Any], shape: [Int]? = nil) {
    //     self.init(BaseNDArray(data, shape: shape))
    // }

    // internal convenience init(_ data: [Scalar], shape: ArrayShape) {
    //     self.init(BaseNDArray(data, shape: shape))
    // }

    // public convenience init(_ data: Scalar) {
    //     self.init(BaseNDArray(data))
    // }
}