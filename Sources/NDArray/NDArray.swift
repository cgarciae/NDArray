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
    public let _shape: Shape

    @inlinable
    public var shape: [Int] { _shape.virtualShape }

    public init(_ data: [Any], shape: [Int]? = nil) {
        (self.data, _shape) = constructArrayAndShape(data, shape)
    }

    public init(_ data: [Scalar], shape: [Int]? = nil) {
        (self.data, _shape) = constructArrayAndShape(data, shape)
    }

    public init(_ data: [Scalar], shape: Shape) {
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
        let indexedDimensions = dimensions
            .enumerated()
            .filter { !($0.1 is SqueezedDimension) }

        for (rangeExpression, index_dimension) in zip(ranges, indexedDimensions) {
            let (i, dimension) = index_dimension

            switch rangeExpression.arrayRange {
            case let .index(index):
                dimensions[i] = dimension.indexed(index)
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
        let indexedDimensions = dimensions
            .enumerated()
            .filter { !($0.1 is SqueezedDimension) }

        for (current, nextIdx) in indexes.enumerated() {
            let trueNext = indexedDimensions[nextIdx].0
            let trueCurrent = indexedDimensions[current].0

            dimensions[trueCurrent] = _shape.dimensions[trueNext]
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