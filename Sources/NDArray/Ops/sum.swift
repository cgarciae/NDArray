
extension NDArray where Scalar: AdditiveArithmetic {
    public func sum(axis: Int) -> NDArray<Scalar> {
        sum(axis: [axis])
    }

    public func sum(axis: [Int]) -> NDArray<Scalar> {
        let outputShape = shape
            .enumerated()
            .filter { !axis.contains($0.offset) }
            .map { $0.element }

        // replace with NDArray(zeros: ...) in the future
        var initial = NDArray(
            [Scalar](
                repeating: Scalar.zero,
                count: outputShape.product()
            ),
            shape: outputShape
        )

        return reduce(axis: axis, initial: &initial, f: +)
    }
}