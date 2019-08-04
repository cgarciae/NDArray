

@usableFromInline internal struct ArrayShape {
    @usableFromInline let dimensions: [DimensionProtocol]
    @usableFromInline let dimensionStrides: [Int]
    // this name might be misleading, its the shape as the user sees it, not its actual memory content
    @usableFromInline let virtualShape: [Int]
    // similar to virtualShape, these are the dimension that the user sees, but there can be more
    // SqueezedDimension that still contain necesary memory_stride information
    @usableFromInline let nonSequeezedDimensions: [(index: Int, dimension: DimensionProtocol)]

    @usableFromInline func realDimension(at index: Int) -> DimensionProtocol {
        nonSequeezedDimensions.first { $0.index == index }!.dimension
    }

    @usableFromInline init(_ shape: [Int]) {
        let dimensionStrides = getDimensionStrides(of: shape)

        let dimensions = (0 ..< shape.count)
            .map { i -> DimensionProtocol in

                if shape[i] == 1 {
                    return SingularDimension()
                } else {
                    return Dimension(length: shape[i], memory_stride: dimensionStrides[i])
                }
            }

        self.init(dimensions)
    }

    @usableFromInline init(_ dimensions: [DimensionProtocol]) {
        self.dimensions = dimensions
        let dimensionLengths = dimensions.map { $0.length }
        dimensionStrides = getDimensionStrides(of: dimensionLengths)
        virtualShape = dimensions.filter { !($0 is SqueezedDimension) }.map { $0.length }
        nonSequeezedDimensions = dimensions
            .lazy
            .enumerated()
            .map { (index: $0.0, dimension: $0.1) }
            .filter { !($0.dimension is SqueezedDimension) }
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        return (0 ..< dimensions.count)
            .reduce((accumulatedIndex: 0, reminder: index)) { state, i in

                let quotient: Int
                let reminder: Int

                if dimensions[i].length == 1 {
                    quotient = 0
                    reminder = state.reminder
                } else {
                    (quotient, reminder) = state.reminder.quotientAndRemainder(dividingBy: dimensionStrides[i])
                }

                let accumulatedIndex = dimensions[i].strideValue(of: quotient) + state.accumulatedIndex

                return (accumulatedIndex: accumulatedIndex, reminder: reminder)
            }
            .accumulatedIndex
    }

    @inlinable
    public subscript(_ indexes: [Int]) -> Int {
        let nonSequeezedValue = zip(indexes, nonSequeezedDimensions)
            .lazy
            .map { index, indexDimension in
                indexDimension.dimension.strideValue(of: index)
            }.reduce(0, +)

        let squeezedValue = dimensions
            .lazy
            .filter { $0 is SqueezedDimension }
            .map { $0.strideValue(of: 0) }
            .reduce(0, +)

        return nonSequeezedValue + squeezedValue
    }
}