

extension NDArray {
    public func reduce<B>(axis: [Int], initial: NDArray<B>, f: (B, Scalar) -> B) -> NDArray<B> {
        var initial = initial
        return reduce(axis: axis, initial: &initial, f: f)
    }

    public func reduce<B>(axis: [Int], initial: inout NDArray<B>, f: (B, Scalar) -> B) -> NDArray<B> {
        if !initial.anyNDArray.isSetable() {
            initial = NDArray<B>(initial.baseCopy())
        }

        let reducingShape = shape
            .enumerated()
            .filter { axis.contains($0.offset) }
            .map { $0.element }

        let nonReducedAxis = Array(
            Set(0 ..< shape.count).subtracting(Set(axis))
        )

        for index in indexSequence(range: 0 ..< reducingShape.product(), shape:reducingShape) {
            var ranges = index.rectangularIndex.map { ArrayRange.index($0) }

            for i in nonReducedAxis.sorted() {
                ranges.insert(.all, at: i)
            }

            let view = self[r: ranges]

            elementwiseAssignApply(&initial, view, apply: f)
        }

        return initial
    }
}