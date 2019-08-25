
extension NDArray where Scalar: AdditiveArithmetic & Comparable {
    public func max(axis: Int) -> NDArray<Scalar> {
        max(axis: [axis])
    }

    public func max(axis: [Int]) -> NDArray<Scalar> {
        let reducingShape = shape
            .enumerated()
            .filter { axis.contains($0.offset) }
            .map { $0.element }

        let nonReducedAxis = Array(
            Set(0 ..< shape.count).subtracting(Set(axis))
        )

        var ranges = reducingShape.map { _ in ArrayRange.index(0) }

        for i in nonReducedAxis.sorted() {
            ranges.insert(.all, at: i)
        }

        // replace with NDArray(zeros: ...) in the future
        var initial = self[r: ranges].copy()

        return reduce(axis: axis, initial: &initial) {
            Swift.max($0, $1)
        }
    }
}