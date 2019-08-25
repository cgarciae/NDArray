

extension NDArray where Scalar: FloatingPoint & Numeric {
    public func mean(axis: Int) -> NDArray<Scalar> {
        mean(axis: [axis])
    }

    public func mean(axis: [Int]) -> NDArray<Scalar> {
        let outputShape = shape
            .enumerated()
            .filter { !axis.contains($0.offset) }
            .map { $0.element }

        let reductionElements = Scalar(shape
            .enumerated()
            .filter { axis.contains($0.offset) }
            .map { $0.element }
            .product()
        )

        // replace with NDArray(zeros: ...) in the future
        var initial = NDArray<Scalar>(
            [Scalar](
                repeating: Scalar.zero,
                count: outputShape.product()
            ),
            shape: outputShape
        )

        return reduce(axis: axis, initial: &initial) { acc, x in
            acc + x / reductionElements
        }
    }
}