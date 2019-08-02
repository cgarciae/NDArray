func constructArrayAndShape<Scalar>(_ data: [Any], _ shape: [Int]? = nil) -> ([Scalar], Shape) {
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

    return (
        flatData,
        Shape(shape ?? calculatedShape)
    )
}

public struct NDArray<Scalar> {
    public let data: [Scalar]
    @usableFromInline internal let _shape: Shape

    @inlinable public var shape: [Int] { _shape.virtualShape }

    public init(_ data: [Any], shape: [Int]? = nil) {
        (self.data, _shape) = constructArrayAndShape(data, shape)
    }

    public init(_ data: [Scalar], shape: [Int]? = nil) {
        (self.data, _shape) = constructArrayAndShape(data, shape)
    }

    @usableFromInline internal init(_ data: [Scalar], shape: Shape) {
        self.data = data
        _shape = shape
    }

    public init(scalar data: Scalar) {
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
        precondition(shape.count >= ranges.count)

        var dimensions = _shape.dimensions

        for (rangeExpression, virtual) in zip(ranges, _shape.nonSequeezedDimensions) {
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

        return NDArray(data, shape: Shape(dimensions))
    }

    @inlinable
    public subscript(_ ranges: ArrayRangeExpression...) -> NDArray {
        self[ranges]
    }

    public func transposed(_ indexes: [Int]) -> NDArray {
        precondition(shape.count >= indexes.count)

        var dimensions = _shape.dimensions

        for (virtualIndexCurrent, virtualIndexNext) in indexes.enumerated() {
            let realIndexCurrent = _shape.nonSequeezedDimensions[virtualIndexCurrent].index

            dimensions[realIndexCurrent] = _shape.nonSequeezedDimensions[virtualIndexNext].dimension
        }

        return NDArray(data, shape: Shape(dimensions))
    }
}

extension NDArray: CustomStringConvertible {
    public var description: String {
        let nElements = shape.reduce(1, *)
        var s = "\(Self.self)\(shape)(" + String(repeating: "[", count: max(shape.count - 1, 0))

        if shape.count == 0 {
            return s + "\(dataValue(at: 0))" + ")"
        } else if shape.count == 1 {
            var arrayString = ""
            for i in 0 ..< nElements {
                arrayString += "\(dataValue(at: i))" + (i + 1 != nElements ? ", " : "")
            }
            return s + "[\(arrayString)])"
        } else {
            let reversedShape = Array(shape.reversed())
            let lastDim = reversedShape[0]
            let secondLastDim = reversedShape[1]

            s += "\n"

            var arrayString = ""

            for i in 0 ... nElements {
                if i % lastDim == 0, i > 0 {
                    s += "    [\(arrayString)],\n"
                    arrayString = ""
                }

                if i % (lastDim * secondLastDim) == 0, i > 0, i < nElements {
                    s += "\n"
                }

                if i < nElements {
                    arrayString += "\(dataValue(at: i))" + ((i + 1) % lastDim != 0 ? ", " : "")
                }
            }
            return s + String(repeating: "]", count: max(shape.count - 1, 1)) + ")"
        }
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