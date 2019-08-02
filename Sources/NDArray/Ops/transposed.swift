extension NDArray {
    public func transposed(_ indexes: [Int]) -> NDArray {
        precondition(shape.count >= indexes.count)

        var dimensions = array_shape.dimensions

        for (virtualIndexCurrent, virtualIndexNext) in indexes.enumerated() {
            let realIndexCurrent = array_shape.nonSequeezedDimensions[virtualIndexCurrent].index

            dimensions[realIndexCurrent] = array_shape.nonSequeezedDimensions[virtualIndexNext].dimension
        }

        return NDArray(data, shape: Shape(dimensions))
    }
}