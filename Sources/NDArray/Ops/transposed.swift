extension BaseNDArray {
    public func transposed(_ indexes: [Int]) -> NDArray<Scalar> {
        precondition(shape.count >= indexes.count)

        return NDArray(BaseNDArray(
            data,
            shape: ArrayShape(
                indexes.map { i in arrayShape.dimensions[i] },
                linearMemoryOffset: arrayShape.linearMemoryOffset
            )
        ))
    }
}