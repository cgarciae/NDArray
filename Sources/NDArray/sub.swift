import func CBlas.cblas_saxpy

extension NDArray where Scalar == Float {
    @inlinable
    public static func - (left: NDArray<Scalar>, right: Scalar) -> NDArray<Scalar> {
        var outputData = [Scalar](repeating: -right, count: left.nElements)

        cblas_saxpy(Int32(left.nElements), 1, left.data, 1, &outputData, 1)

        return NDArray(
            outputData,
            shape: left.shape
        )
    }

    @inlinable
    public static func - (left: Scalar, right: NDArray<Scalar>) -> NDArray<Scalar> {
        right - left
    }

    @inlinable
    public static func - (left: NDArray<Scalar>, right: NDArray<Scalar>) -> NDArray<Scalar> {
        precondition(left.shape == right.shape)

        var outputData = Array(left.data)

        cblas_saxpy(Int32(left.nElements), -1, right.data, 1, &outputData, 1)

        return NDArray(
            outputData,
            shape: left.shape
        )
    }
}

internal extension NDArray where Scalar == Float {
    @inlinable
    static func _vjpSubtract(lhs: NDArray, rhs: NDArray) ->
        (
            NDArray,
            (NDArray) -> (NDArray, NDArray)
        ) {
        let value = lhs - rhs
        return (
            value,
            { dvalue in (dvalue * lhs, -1 * dvalue * rhs) }
        )
    }
}
