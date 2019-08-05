extension NDArray {
    public func transposed(_ indexes: [Int]) -> NDArray {
        precondition(shape.count >= indexes.count)

        return NDArray(
            data,
            shape: ArrayShape(
                indexes.map { i in arrayShape.dimensions[i] },
                linearMemoryOffset: arrayShape.linearMemoryOffset
            )
        )
    }
}