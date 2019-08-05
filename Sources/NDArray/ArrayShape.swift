

@usableFromInline internal struct ArrayShape {
    @usableFromInline let dimensions: [DimensionProtocol]
    @usableFromInline let linearMemoryOffset: Int
    @usableFromInline let dimensionLengths: [Int]
    @usableFromInline let dimensionStrides: [Int]

    @usableFromInline init(_ shape: [Int]) {
        let dimensionStrides = getDimensionStrides(of: shape)
        self.dimensionStrides = dimensionStrides

        dimensionLengths = shape
        linearMemoryOffset = 0

        dimensions = (0 ..< shape.count)
            .map { i -> DimensionProtocol in

                if shape[i] == 1 {
                    return SingularDimension()
                } else {
                    return Dimension(length: shape[i], memory_stride: dimensionStrides[i])
                }
            }
    }

    @usableFromInline init(_ dimensions: [DimensionProtocol], linearMemoryOffset: Int) {
        self.dimensions = dimensions
        self.linearMemoryOffset = linearMemoryOffset
        dimensionLengths = dimensions.map { $0.length }
        dimensionStrides = getDimensionStrides(of: dimensionLengths)
    }

    @inlinable
    public func linearIndex(of indexes: [Int]) -> Int {
        let partialIndex = zip(indexes, dimensions)
            .lazy
            .map { index, dimension in
                dimension.strideValue(of: index)
            }
            .sum()

        return partialIndex + linearMemoryOffset
    }
}