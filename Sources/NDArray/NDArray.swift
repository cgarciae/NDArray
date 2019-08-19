

public protocol NDArrayProtocol {
    associatedtype Scalar

    typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    var shape: [Int] { get }

    func subscript_get(_: [ArrayRange]) -> NDArray<Scalar>
    mutating func subscript_set(_: [ArrayRange], _: NDArray<Scalar>) -> Void
    func linearIndex(at indexes: [Int]) -> Int
    func dataValue(at indexes: [Int]) -> Scalar
    func withScalarGetter(_: (ScalarGetter) -> Void)
    func withScalarSetter(_: (ScalarSetter) -> Void)

    // ops
    func transposed(_ indexes: [Int]) -> NDArray<Scalar>
}

public struct NDArray<Scalar>: NDArrayProtocol {
    public let shape: [Int]

    public typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    public typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    let _linearIndex: ([Int]) -> Int
    let _dataValue: ([Int]) -> Scalar
    let _subscript_get: ([ArrayRange]) -> NDArray<Scalar>
    let _subscript_set: ([ArrayRange], NDArray<Scalar>) -> Void
    let _withScalarGetter: ((ScalarGetter) -> Void) -> Void
    let _withScalarSetter: ((ScalarSetter) -> Void) -> Void

    public init<N:NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        var ndarray = ndarray

        shape = ndarray.shape
        _linearIndex = ndarray.linearIndex
        _dataValue = ndarray.dataValue
        _subscript_get = ndarray.subscript_get
        _subscript_set = { ndarray.subscript_set($0, $1) }
        _withScalarGetter = ndarray.withScalarGetter
        _withScalarSetter = ndarray.withScalarSetter
    }

    init(_ ndarray: NDArray) {
        self = ndarray
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

    public func withScalarGetter(_ body: (ScalarGetter) -> Void) {
        _withScalarGetter(body)
    }

    public func withScalarSetter(_ body: (ScalarSetter) -> Void) {
        _withScalarSetter(body)
    }
}