public protocol NDArrayFloatingPoint: Numeric & Divisible
// :
// BinaryFloatingPoint & Differentiable & ElementaryFunctions
// where Self.RawSignificand: FixedWidthInteger,
// Self == Self.TangentVector,
// Self == Self.AllDifferentiableVariables
{}

extension Float: NDArrayFloatingPoint {}
extension Double: NDArrayFloatingPoint {}