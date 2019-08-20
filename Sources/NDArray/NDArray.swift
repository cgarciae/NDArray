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
    func withScalarSetter(_: (@escaping ScalarSetter) -> Void)

    // ops
    func transposed(_: [Int]) -> NDArray<Scalar>
    func tiled(by: [Int]) -> NDArray<Scalar>
    func expandDimensions(axis: Int) -> NDArray<Scalar>
    func scalarized() -> Scalar

    //
    func toArray<T: MultiArray>(_: T.Type) -> T

    //
    mutating func copyInternals() -> Void
}

extension NDArrayProtocol {
    public func toArray<T: MultiArray>(_: T.Type) -> T {
        let cp: BaseNDArray = copy()
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
    let _copyInternals: () -> Void

    public init<N:NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        var ndarray_ref = Ref(ndarray)

        shape = ndarray_ref.value.shape
        _linearIndex = { ndarray_ref.value.linearIndex(at: $0) }
        _dataValue = { ndarray_ref.value.dataValue(at: $0) }
        _subscript_get = { ndarray_ref.value.subscript_get($0) }
        _subscript_set = {
            if !isKnownUniquelyReferenced(&ndarray_ref) {
                ndarray_ref.value.copyInternals()
                ndarray_ref = Ref(ndarray)
            }
            ndarray_ref.value.subscript_set($0, $1)
        }
        _withScalarGetter = { ndarray_ref.value.withScalarGetter($0) }
        _withScalarSetter = { ndarray_ref.value.withScalarSetter($0) }
        _transposed = { ndarray_ref.value.transposed($0) }
        _tiled = { ndarray_ref.value.tiled(by: $0) }
        _expandDimensions = { ndarray_ref.value.expandDimensions(axis: $0) }
        _scalarized = { ndarray_ref.value.scalarized() }
        _copyInternals = { ndarray_ref.value.copyInternals() }
    }

    init(_ ndarray: NDArray) {
        self = ndarray
    }

    public init(_ data: [Any], shape: [Int]? = nil) {
        self = NDArray(BaseNDArray(data, shape: shape))
    }

    internal init(_ data: [Scalar], shape: ArrayShape) {
        self = NDArray(BaseNDArray(data, shape: shape))
    }

    public init(_ data: Scalar) {
        self = NDArray(BaseNDArray(data))
    }

    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        _subscript_get(ranges)
    }

    public func subscript_set(_ ranges: [ArrayRange], _ value: NDArray<Scalar>) {
        _subscript_set(ranges, value)
    }

    public func linearIndex(at indexes: [Int]) -> Int {
        _linearIndex(indexes)
    }

    public func dataValue(at indexes: [Int]) -> Scalar {
        _dataValue(indexes)
    }

    public func withScalarGetter(_ body: (@escaping ScalarGetter) -> Void) {
        _withScalarGetter(body)
    }

    public func withScalarSetter(_ body: (@escaping ScalarSetter) -> Void) {
        _withScalarSetter(body)
    }

    public func transposed(_ permutations: [Int]) -> NDArray<Scalar> {
        _transposed(permutations)
    }

    public func tiled(by amounts: [Int]) -> NDArray<Scalar> {
        _tiled(amounts)
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        _expandDimensions(axis)
    }

    public func scalarized() -> Scalar {
        _scalarized()
    }

    public mutating func copyInternals() {
        _copyInternals()
    }
}