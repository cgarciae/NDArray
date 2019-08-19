

protocol NDArrayProtocol {
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
}

public struct NDArray<Scalar>: NDArrayProtocol {
    let shape: [Int]

    typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void

    let _linearIndex: ([Int]) -> Int
    let _dataValue: ([Int]) -> Scalar
    let _subscript_get: ([ArrayRange]) -> NDArray<Scalar>
    let _subscript_set: ([ArrayRange], NDArray<Scalar>) -> Void
    let _withScalarGetter: ((ScalarGetter) -> Void) -> Void
    let _withScalarSetter: ((ScalarSetter) -> Void) -> Void

    func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        _subscript_get(ranges)
    }

    func subscript_set(_ ranges: [ArrayRange], _ value: NDArray<Scalar>) {
        _subscript_set(ranges, value)
    }

    func linearIndex(at indexes: [Int]) -> Int {
        _linearIndex(indexes)
    }

    func dataValue(at indexes: [Int]) -> Scalar {
        _dataValue(indexes)
    }

    func withScalarGetter(_ body: (ScalarGetter) -> Void) {
        _withScalarGetter(body)
    }

    func withScalarSetter(_ body: (ScalarSetter) -> Void) {
        _withScalarSetter(body)
    }
}