

public struct NDArray<Scalar> {
    public var data: [Scalar]
    public let shape: [Int]

    public var nElements: Int {
        data.count
    }

    public init(_ data: [Scalar], shape: [Int]) {
        precondition(shape.reduce(1, *) == data.count)

        self.data = data
        self.shape = shape
    }

    public init(_ data: Scalar) {
        self.data = [data]
        shape = []
    }
}

// //===------------------------------------------------------------------------------------------===//
// // Equatable
// //===------------------------------------------------------------------------------------------===//
// extension NDArray: Equatable where Scalar: Equatable {
//     @inlinable
//     public static func == (lhs: NDArray, rhs: NDArray) -> Bool {
//         // TODO: This is not correct due to broadcasting.
//         return lhs.data == rhs.data
//     }

//     @inlinable
//     public static func != (lhs: NDArray, rhs: NDArray) -> Bool {
//         // TODO: This is not correct due to broadcasting.
//         return lhs.data != rhs.data
//     }
// }

// //===------------------------------------------------------------------------------------------===//
// // Additive Group
// //===------------------------------------------------------------------------------------------===//
// extension NDArray: AdditiveArithmetic where Scalar == Float {
//     /// The scalar zero tensor.
//     @inlinable
//     public static var zero: NDArray { NDArray(0) }

//     /// Adds two tensors and produces their sum.
//     /// - Note: `+` supports broadcasting.
//     // @inlinable
//     // @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar == Float)
//     // public static func + (lhs: NDArray, rhs: NDArray) -> NDArray {
//     //     return lhs + rhs
//     // }

//     /// Subtracts one tensor from another and produces their difference.
//     /// - Note: `-` supports broadcasting.
//     // @inlinable
//     // @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar == Float)
//     // public static func - (lhs: NDArray, rhs: NDArray) -> NDArray {
//     //     return lhs - rhs
//     // }
// }

// //===------------------------------------------------------------------------------------------===//
// // Multiplicative Group
// //===------------------------------------------------------------------------------------------===//
// extension NDArray: PointwiseMultiplicative where Scalar == Float {
//     /// The scalar one tensor.
//     @inlinable
//     public static var one: NDArray { NDArray(1) }

//     /// Returns the element-wise reciprocal of `self`.
//     @inlinable
//     public var reciprocal: NDArray { 1 / self }

//     /// Multiplies two tensors element-wise and produces their product.
//     /// - Note: `.*` supports broadcasting.
//     public static func .* (lhs: NDArray, rhs: NDArray) -> NDArray {
//         return lhs * rhs
//     }
// }

// //===------------------------------------------------------------------------------------------===//
// // Differentiable
// //===------------------------------------------------------------------------------------------===//

// // extension NDArray: Differentiable where Scalar == Float {
// //     public typealias TangentVector = NDArray
// //     public typealias AllDifferentiableVariables = NDArray
// // }
