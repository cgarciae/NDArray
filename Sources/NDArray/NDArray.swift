

public struct NDArray<Scalar> {
    public let data: [Scalar]
    public let _shape: Shape

    @inlinable
    public var shape: [Int] { _shape.virtualShape }

    public init(_ data: [Any], shape: [Int]? = nil) {
        // precondition(shape.reduce(1, *) == data.count)
        let (flatData, calculatedShape): ([Scalar], [Int]) = flattenArrays(data)

        precondition(
            calculatedShape.reduce(1, *) == flatData.count,
            "All sub-arrays in data must have equal length. Calculated shape: \(calculatedShape), \(flatData)"
        )

        if let shape = shape {
            precondition(
                shape.reduce(1, *) == flatData.count,
                "Invalid shape, number of elements"
            )
        }

        let shape = shape ?? calculatedShape

        self.data = flatData
        _shape = Shape(shape)
    }

    public init(_ data: [Scalar], shape: [Int]) {
        precondition(shape.reduce(1, *) == data.count)

        self.data = data
        _shape = Shape(shape)
    }

    public init(_ data: [Scalar], shape: Shape) {
        // precondition(shape.dimensionLengths.reduce(1, *) == data.count)

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
    public func dataValue(at index: Int) -> Scalar {
        let realIndex = _shape.realIndex(of: index)
        return data[realIndex]
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