

public struct NDArray<Scalar> {
    public let data: [Scalar]
    public let _shape: Shape

    @inlinable
    public var shape: [Int] { _shape.dimension_lengths }

    public init(_ data: [Scalar], shape: [Int]) {
        // precondition(shape.reduce(1, *) == data.count)

        self.data = data
        _shape = Shape(shape)
    }

    public init(_ data: [Scalar], shape: Shape) {
        // precondition(shape.dimension_lengths.reduce(1, *) == data.count)

        self.data = data
        _shape = shape
    }

    public init(_ data: Scalar) {
        self.data = [data]
        _shape = Shape([DimensionProtocol]())
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        _shape.realIndex(of: index)
    }

    @inlinable
    public subscript(_ ranges: [ArrayRangeExpression]) -> NDArray {
        let dimensions = zip(ranges, _shape.dimensions)
            .map { rangeExpression, dimension -> DimensionProtocol in
                switch rangeExpression.arrayRange {
                case let .index(index):
                    return dimension.indexed(index)
                }
            }

        return NDArray(data, shape: Shape(dimensions))
    }

    @inlinable
    public subscript(_ ranges: ArrayRangeExpression...) -> NDArray {
        self[ranges]
    }
}

public protocol ArrayRangeExpression {
    @inlinable
    var arrayRange: ArrayRange { get }
}

public enum ArrayRange: ArrayRangeExpression {
    case index(Int)
    // case ellipsis
    // case newAxis
    // case squeezeAxis
    // case range(Range<Int>, stride: Int)
    // case closedRange(ClosedRange<Int>, stride: Int)
    // case partialRangeFrom(PartialRangeFrom<Int>, stride: Int)
    // case partialRangeUpTo(PartialRangeUpTo<Int>, stride: Int)
    // case partialRangeThrough(PartialRangeThrough<Int>, stride: Int)

    public var arrayRange: ArrayRange { self }
}

extension Int: ArrayRangeExpression {
    public var arrayRange: ArrayRange { .index(self) }
}