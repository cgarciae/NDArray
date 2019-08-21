public struct ScalarNDArray<Scalar>: NDArrayProtocol {
    public let scalar: Scalar
    public let shape: [Int]
    public func subscript_get(_: [ArrayRange]) -> NDArray<Scalar> {
        NDArray(self)
    }

    public mutating func subscript_set(_: [ArrayRange], _: NDArray<Scalar>) {}
    public func linearIndex(at indexes: [Int]) -> Int {}
    public func dataValue(at indexes: [Int]) -> Scalar {}
    public func withScalarGetter(_: (@escaping NDArray<Scalar>.ScalarGetter) -> Void) {}
    public mutating func withScalarSetter(_: (@escaping NDArray<Scalar>.ScalarSetter) -> Void) {}

    // ops
    public func transposed(_: [Int]) -> NDArray<Scalar> {}
    public func tiled(by: [Int]) -> NDArray<Scalar> {}
    public func expandDimensions(axis: Int) -> NDArray<Scalar> {}
    public func scalarized() -> Scalar {}

    //
    public func toArray<T: MultiArray>(_: T.Type) -> T {}

    //
    public func copy() -> Self {}
}