import func CBlas.cblas_saxpy
import func CBlas.cblas_sscal

// extension NDArray where Scalar == Float {
//     public static func / (_ left: NDArray<Scalar>, _ right: Scalar) -> NDArray<Scalar> {
//         let right = 1 / right
//         var outputData = Array(left.data)

//         cblas_sscal(Int32(left.data.count), right, &outputData, 1)

//         return NDArray(
//             outputData,
//             shape: left.shape
//         )
//     }

//     public static func / (_ left: Scalar, _ right: NDArray<Scalar>) -> NDArray<Scalar> {
//         let left = NDArray(
//             Array(repeating: left, count: right.shape.reduce(1, *)),
//             shape: right.shape
//         )

//         return left / right
//     }

//     public static func / (_ left: NDArray<Scalar>, _ right: NDArray<Scalar>) -> NDArray<Scalar> {
//         precondition(left.shape == right.shape)

//         return elementWise(
//             left,
//             right,
//             apply: /
//         )
//     }
// }