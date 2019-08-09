
@usableFromInline
internal func broadcast(_ left: ArrayShape, and right: ArrayShape) -> (left: ArrayShape, right: ArrayShape) {
    var left = left
    var right = right

    if left.dimensions.count == 0 {
        left = ArrayShape(
            Array(repeating: SingularDimension(), count: right.dimensions.count),
            linearMemoryOffset: left.linearMemoryOffset
        )
    } else if right.dimensions.count == 0 {
        right = ArrayShape(
            Array(repeating: SingularDimension(), count: left.dimensions.count),
            linearMemoryOffset: right.linearMemoryOffset
        )
    }

    var leftDimensions = left.dimensions
    var rightDimensions = right.dimensions

    for i in 0 ..< leftDimensions.count {
        if leftDimensions[i].length == rightDimensions[i].length {
            continue
        } else if leftDimensions[i].length == 1 {
            leftDimensions[i] = leftDimensions[i].tiled(rightDimensions[i].length)
        } else {
            rightDimensions[i] = rightDimensions[i].tiled(leftDimensions[i].length)
        }
    }

    return (
        left: ArrayShape(leftDimensions, linearMemoryOffset: left.linearMemoryOffset),
        right: ArrayShape(rightDimensions, linearMemoryOffset: right.linearMemoryOffset)
    )
}

@usableFromInline
internal func broadcast<A, B>(_ left: NDArray<A>, and right: NDArray<B>) -> (left: NDArray<A>, right: NDArray<B>) {
    precondition(
        left.shape == [] || right.shape == [] || zip(left.shape, right.shape).map { leftLength, rightLength in
            leftLength == rightLength || leftLength == 1 || rightLength == 1
        }.reduce(true) { $0 && $1 },
        "Cannot broadcast shapes \(left.shape) and \(right.shape)"
    )

    let (leftShape, rightShape) = broadcast(left.arrayShape, and: right.arrayShape)

    return (
        left: NDArray(left.data, shape: leftShape),
        right: NDArray(right.data, shape: rightShape)
    )
}