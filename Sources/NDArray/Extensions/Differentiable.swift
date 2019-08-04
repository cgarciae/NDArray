

#if canImport(TensorFlow)
    public protocol NDArrayFloatingPoint:
        BinaryFloatingPoint & Differentiable & ElementaryFunctions
        where Self.RawSignificand: FixedWidthInteger,
        Self == Self.TangentVector,
        Self == Self.AllDifferentiableVariables {}

    extension Float: NDArrayFloatingPoint {}
    extension Double: NDArrayFloatingPoint {}

    //===------------------------------------------------------------------------------------------===//
    // Equatable
    //===------------------------------------------------------------------------------------------===//
    extension NDArray: Equatable where Scalar: Equatable {
        @inlinable
        public static func == (lhs: NDArray, rhs: NDArray) -> Bool {
            // TODO: This is not correct due to broadcasting.
            elementwise(lhs, rhs, apply: ==).data.value.reduce(true) { $0 && $1 }
        }

        @inlinable
        public static func != (lhs: NDArray, rhs: NDArray) -> Bool {
            // TODO: This is not correct due to broadcasting.
            !(lhs == rhs)
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // Additive Group
    //===------------------------------------------------------------------------------------------===//
    extension NDArray: AdditiveArithmetic where Scalar: Numeric {
        /// The scalar zero tensor.
        @inlinable
        public static var zero: NDArray { NDArray(0) }

        /// Adds two tensors and produces their sum.
        /// - Note: `+` supports broadcasting.
        @inlinable
        @differentiable(vjp: _vjpAdd(lhs:rhs:) where Scalar: NDArrayFloatingPoint)
        public static func + (lhs: NDArray, rhs: NDArray) -> NDArray {
            elementwise(lhs, rhs, apply: +)
        }

        /// Subtracts one tensor from another and produces their difference.
        /// - Note: `-` supports broadcasting.
        @inlinable
        @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Scalar: NDArrayFloatingPoint)
        public static func - (lhs: NDArray, rhs: NDArray) -> NDArray {
            elementwise(lhs, rhs, apply: -)
        }
    }

    internal extension NDArray where Scalar: NDArrayFloatingPoint {
        @inlinable
        static func _vjpAdd(
            lhs: NDArray,
            rhs: NDArray
        ) -> (NDArray, (NDArray) -> (NDArray, NDArray)) {
            return (lhs + rhs, { [lhs, rhs] v in
                (
                    lhs * v,
                    rhs * v
                )
            })
        }

        @inlinable
        static func _vjpSubtract(
            lhs: NDArray,
            rhs: NDArray
        ) -> (NDArray, (NDArray) -> (NDArray, NDArray)) {
            return (lhs - rhs, { [lhs, rhs] v in
                (
                    lhs * v,
                    NDArray(-1) * rhs * v
                )
            })
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // Multiplicative Group
    //===------------------------------------------------------------------------------------------===//
    extension NDArray: PointwiseMultiplicative where Scalar: Numeric & Divisible {
        /// The scalar one tensor.
        @inlinable
        public static var one: NDArray { NDArray(1) }

        /// Returns the element-wise reciprocal of `self`.
        @inlinable
        public var reciprocal: NDArray { 1 / self }

        /// Multiplies two tensors element-wise and produces their product.
        /// - Note: `.*` supports broadcasting.
        public static func .* (lhs: NDArray, rhs: NDArray) -> NDArray {
            return lhs * rhs
        }
    }

    //===------------------------------------------------------------------------------------------===//
    // Differentiable
    //===------------------------------------------------------------------------------------------===//
    extension NDArray: Differentiable where Scalar: NDArrayFloatingPoint {
        public typealias TangentVector = NDArray
        public typealias AllDifferentiableVariables = NDArray
    }

    extension NDArray where Scalar: NDArrayFloatingPoint {
        // @differentiable(vjp: _vpjSum(_:) where Scalar: NDArrayFloatingPoint)
        public func sum() -> Scalar {
            copy().data.value.reduce(0, +)
        }

        @differentiating(sum)
        public func _vpjSum() -> (value: Scalar, pullback: (Scalar) -> NDArray) {
            (
                value: sum(),
                pullback: { dy in
                    NDArray(
                        Array(repeating: dy, count: self.shape.reduce(1, *)),
                        shape: self.shape
                    )
                }
            )
        }
    }

#endif