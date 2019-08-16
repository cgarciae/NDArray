
extension NDArray where Scalar: AdditiveArithmetic {
    public static func + (lhs: NDArray<Scalar>, rhs: NDArray<Scalar>) -> NDArray<Scalar> {
        elementwise(lhs, rhs, apply: +)
    }

    public static func + (lhs: NDArray<Scalar>, rhs: Scalar) -> NDArray<Scalar> {
        elementwise(lhs) { $0 + rhs }
    }

    public static func + (lhs: Scalar, rhs: NDArray<Scalar>) -> NDArray<Scalar> {
        elementwise(rhs) { lhs + $0 }
    }
}

// import func CBlas.cblas_saxpy

// extension NDArray where Scalar: AdditiveArithmetic {
//     @inlinable
//     public static func + (left: NDArray, right: Scalar) -> NDArray<Float> {
//         var outputData = [Scalar](repeating: right, count: left.data.count)

//         cblas_saxpy(Int32(left.data.count), 1, left.data, 1, &outputData, 1)

//         return NDArray(
//             outputData,
//             shape: left.shape
//         )
//     }

//     @inlinable
//     public static func + (left: Scalar, right: NDArray<Scalar>) -> NDArray<Scalar> {
//         right + left
//     }

//     @inlinable
//     public static func + (left: NDArray<Scalar>, right: NDArray<Scalar>) -> NDArray<Scalar> {
//         precondition(left.shape == right.shape)

//         var outputData = Array(right.data)

//         cblas_saxpy(Int32(left.data.count), 1, left.data, 1, &outputData, 1)

//         return NDArray(
//             outputData,
//             shape: left.shape
//         )
//     }
// }

// extension NDArray where Scalar == Float {
//     @inlinable
//     public static func + (left: NDArray, right: Scalar) -> NDArray<Float> {
//         var outputData = [Scalar](repeating: right, count: left.data.count)

//         cblas_saxpy(Int32(left.data.count), 1, left.data, 1, &outputData, 1)

//         return NDArray(
//             outputData,
//             shape: left.shape
//         )
//     }

//     @inlinable
//     public static func + (left: NDArray<Scalar>, right: NDArray<Scalar>) -> NDArray<Scalar> {
//         precondition(left.shape == right.shape)

//         var outputData = Array(right.data)

//         cblas_saxpy(Int32(left.data.count), 1, left.data, 1, &outputData, 1)

//         return NDArray(
//             outputData,
//             shape: left.shape
//         )
//     }
// }