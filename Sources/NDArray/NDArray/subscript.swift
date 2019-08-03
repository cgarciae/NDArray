
extension NDArray {
    @inlinable
    public subscript(_ ranges: [ArrayRangeExpression]) -> NDArray {
        precondition(shape.count >= ranges.count)

        var dimensions = array_shape.dimensions

        for (rangeExpression, virtual) in zip(ranges, array_shape.nonSequeezedDimensions) {
            switch rangeExpression.arrayRange {
            case let .index(index):
                dimensions[virtual.index] = virtual.dimension.indexed(index)

            case let .range(range, stride: stride):

                dimensions[virtual.index] = virtual.dimension.sliced(
                    start: range.lowerBound,
                    end: range.upperBound,
                    stride: stride
                )

            case let .closedRange(range, stride: stride):

                dimensions[virtual.index] = virtual.dimension.sliced(
                    start: range.lowerBound,
                    end: range.upperBound + 1,
                    stride: stride
                )

            case let .partialRangeUpTo(range, stride: stride):
                dimensions[virtual.index] = virtual.dimension.sliced(
                    start: 0,
                    end: range.upperBound,
                    stride: stride
                )

            case let .partialRangeThrough(range, stride: stride):

                dimensions[virtual.index] = virtual.dimension.sliced(
                    start: 0,
                    end: range.upperBound + 1,
                    stride: stride
                )

            case let .partialRangeFrom(range, stride: stride):

                dimensions[virtual.index] = virtual.dimension.sliced(
                    start: range.lowerBound,
                    stride: stride
                )
            }
        }

        return NDArray(data, shape: ArrayShape(dimensions))
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
    case range(Range<Int>, stride: Int)
    case closedRange(ClosedRange<Int>, stride: Int)
    case partialRangeFrom(PartialRangeFrom<Int>, stride: Int)
    case partialRangeUpTo(PartialRangeUpTo<Int>, stride: Int)
    case partialRangeThrough(PartialRangeThrough<Int>, stride: Int)

    public var arrayRange: ArrayRange { self }
}

extension Int: ArrayRangeExpression {
    public var arrayRange: ArrayRange { .index(self) }
}

extension Range: ArrayRangeExpression where Bound == Int {
    public var arrayRange: ArrayRange { .range(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .range(self, stride: stride)
    }
}

extension ClosedRange: ArrayRangeExpression where Bound == Int {
    public var arrayRange: ArrayRange { .closedRange(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .closedRange(self, stride: stride)
    }
}

extension PartialRangeFrom: ArrayRangeExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeFrom(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeFrom(self, stride: stride)
    }
}

extension PartialRangeUpTo: ArrayRangeExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeUpTo(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeUpTo(self, stride: stride)
    }
}

extension PartialRangeThrough: ArrayRangeExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeThrough(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeThrough(self, stride: stride)
    }
}