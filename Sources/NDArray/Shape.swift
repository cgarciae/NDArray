
public protocol ShapeProtocol {
    // func indexIterator() -> AnyIterator<Int>
}

public struct Shape: ShapeProtocol {
    public let dimensions: [DimensionProtocol]
    public let dimensionLengths: [Int]
    public let dimensionStrides: [Int]
    public let virtualShape: [Int]

    public init(_ shape: [Int]) {
        let dimensionStrides = getDimensionStrides(of: shape)

        dimensions = (0 ..< shape.count)
            .map { i -> DimensionProtocol in

                if shape[i] == 1 {
                    return SingularDimension()
                } else {
                    return Dimension(length: shape[i], memory_stride: dimensionStrides[i])
                }
            }

        dimensionLengths = dimensions.map { $0.length }
        virtualShape = dimensionLengths
        self.dimensionStrides = dimensionStrides
    }

    public init(_ dimensions: [DimensionProtocol]) {
        self.dimensions = dimensions
        dimensionLengths = dimensions.map { $0.length }
        dimensionStrides = getDimensionStrides(of: dimensionLengths)
        virtualShape = dimensions.filter { !($0 is SqueezedDimension) }.map { $0.length }
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

                let accumulatedIndex = dimensions[i].memoryStridedValue(of: quotient) + state.accumulatedIndex

                return (accumulatedIndex: accumulatedIndex, reminder: reminder)
            }
            .accumulatedIndex
    }

    @inlinable
    public func realIndex(of indexes: [Int]) -> Int {
        precondition(
            indexes.count == dimensionLengths.count,
            "Invalid index dimensions, expected \(dimensionLengths.count), got \(indexes.count)"
        )
        precondition(
            zip(indexes, dimensionLengths).map(<=).reduce(true) { $0 && $1 },
            "Index out of bounds, expected values in the range of \(dimensionLengths), got \(indexes)"
        )

        return zip(dimensions, indexes)
            .lazy
            .map { dimension, index in
                dimension.memoryStridedValue(of: index)
            }
            .reduce(0, +)
    }

    @inlinable
    public subscript(_ indexes: Int...) -> Int {
        realIndex(of: indexes)
    }
}