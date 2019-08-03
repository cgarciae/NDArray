extension NDArray {
    public func transposed(_ indexes: [Int]) -> NDArray {
        precondition(shape.count >= indexes.count)

        var dimensions = arrayShape.dimensions

        for (virtualIndexCurrent, virtualIndexNext) in indexes.enumerated() {
            let realIndexCurrent = arrayShape.nonSequeezedDimensions[virtualIndexCurrent].index

            dimensions[realIndexCurrent] = arrayShape.nonSequeezedDimensions[virtualIndexNext].dimension
        }

        return NDArray(data, shape: ArrayShape(dimensions))
    }
}