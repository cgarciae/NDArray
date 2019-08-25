

public protocol NDArrayProtocol {
    associatedtype Scalar

    typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void
    typealias ScalarGetterSetter = ((Int, UnsafeMutableBufferPointer<Int>), (Scalar) -> Scalar) -> Void

    var shape: [Int] { get }

    func subscript_get(_: [ArrayRange]) -> NDArray<Scalar>
    mutating func subscript_set(_: [ArrayRange], _: NDArray<Scalar>) -> NDArray<Scalar>
    func linearIndex(at indexes: [Int]) -> Int
    func dataValue(at indexes: [Int]) -> Scalar
    func withScalarGetter(_: (@escaping ScalarGetter) -> Void)
    mutating func withScalarSetter(_: (@escaping ScalarSetter) -> Void)
    mutating func withScalarGetterSetter(_: (@escaping ScalarGetterSetter) -> Void)

    // ops
    func transposed(_: [Int]) -> NDArray<Scalar>
    func tiled(by: [Int]) -> NDArray<Scalar>
    func expandDimensions(axis: Int) -> NDArray<Scalar>
    func scalarized() -> Scalar

    //
    // mutating func copyInternals() -> Void
    func copy() -> NDArray<Scalar>
}

public protocol MultiArray {
    associatedtype Element

    var array: [Element] { get }
}

extension Array : MultiArray {
    public var array: [Element] { self }
}

public protocol SetableNDArray {}