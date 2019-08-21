
@usableFromInline
internal func broadcast<A, B>(_ left: NDArray<A>, and right: NDArray<B>) -> (left: NDArray<A>, right: NDArray<B>) {
    if left.shape == right.shape {
        return (
            left: left,
            right: right
        )
    }

    var left = left
    var right = right

    precondition(
        left.shape == [] || right.shape == [] || zip(left.shape, right.shape).map { leftLength, rightLength in
            leftLength == rightLength || leftLength == 1 || rightLength == 1
        }.reduce(true) { $0 && $1 },
        "Cannot broadcast shapes \(left.shape) and \(right.shape)"
    )

    if left.shape.count == 0 {
        for _ in 0 ..< right.shape.count {
            left = left.expandDimensions(axis: 0)
        }
    } else if right.shape.count == 0 {
        for _ in 0 ..< left.shape.count {
            right = right.expandDimensions(axis: 0)
        }
    }

    var leftRepetitions = Array(repeating: 1, count: left.shape.count)
    var rightRepetitions = Array(repeating: 1, count: left.shape.count)

    for i in 0 ..< left.shape.count {
        if left.shape[i] == right.shape[i] {
            continue
        } else if left.shape[i] == 1 {
            leftRepetitions[i] = right.shape[i]
        } else {
            rightRepetitions[i] = left.shape[i]
        }
    }

    left = left.tiled(by: leftRepetitions)
    right = right.tiled(by: rightRepetitions)

    return (
        left: left,
        right: right
    )
}