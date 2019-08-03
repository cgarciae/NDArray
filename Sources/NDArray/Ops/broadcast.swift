
@usableFromInline
internal func broadcast(_ left: ArrayShape, and right: ArrayShape) -> (left: ArrayShape, right: ArrayShape) {
    var leftDimensions = left.dimensions
    var rightDimensions = right.dimensions

    for (leftIndexDimension, rightIndexDimension) in zip(left.nonSequeezedDimensions, right.nonSequeezedDimensions) {
        if leftIndexDimension.dimension.length == rightIndexDimension.dimension.length {
            continue
        } else if leftIndexDimension.dimension.length == 1 {
            leftDimensions[leftIndexDimension.index] = leftIndexDimension.dimension.tiled(rightIndexDimension.dimension.length)
        } else {
            rightDimensions[rightIndexDimension.index] = rightIndexDimension.dimension.tiled(leftIndexDimension.dimension.length)
        }
    }

    return (
        left: ArrayShape(leftDimensions),
        right: ArrayShape(rightDimensions)
    )
}

@usableFromInline
internal func broadcast<A, B>(_ left: NDArray<A>, and right: NDArray<B>) -> (left: NDArray<A>, right: NDArray<B>) {
    precondition(
        zip(left.shape, right.shape).map { leftLength, rightLength in
            leftLength == rightLength || leftLength == 1 || rightLength == 1
        }.reduce(true) { $0 && $1 },
        "Cannot broadcast shapes \(left.shape) and \(right.shape)"
    )

    let (leftShape, rightShape) = broadcast(left.array_shape, and: right.array_shape)

    return (
        left: NDArray(left.data, shape: leftShape),
        right: NDArray(right.data, shape: rightShape)
    )
}