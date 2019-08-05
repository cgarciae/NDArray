import Foundation

extension NDArray {
    @inlinable
    public subscript(_ ranges: [ArrayRangeExpression]) -> NDArray {
        get {
            precondition(shape.count >= ranges.count)

            var dimensions = arrayShape.dimensions
            var linearMemoryOffset = arrayShape.linearMemoryOffset
            var dimensionToBeRemoved = [Int]()

            for (i, rangeExpression) in ranges.enumerated() {
                switch rangeExpression.arrayRange {
                case let .index(index):
                    linearMemoryOffset += dimensions[i].strideValue(of: index)
                    dimensionToBeRemoved.append(i)

                case let .range(range, stride: stride):

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        end: range.upperBound,
                        stride: stride
                    )

                case let .closedRange(range, stride: stride):

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        end: range.upperBound + 1,
                        stride: stride
                    )

                case let .partialRangeUpTo(range, stride: stride):
                    dimensions[i] = dimensions[i].sliced(
                        start: 0,
                        end: range.upperBound,
                        stride: stride
                    )

                case let .partialRangeThrough(range, stride: stride):

                    dimensions[i] = dimensions[i].sliced(
                        start: 0,
                        end: range.upperBound + 1,
                        stride: stride
                    )

                case let .partialRangeFrom(range, stride: stride):

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        stride: stride
                    )
                }
            }

            dimensions = dimensions
                .enumerated()
                .filter { i, d in !dimensionToBeRemoved.contains(i) }
                .map { i, d in d }

            return NDArray(
                data,
                shape: ArrayShape(
                    dimensions,
                    linearMemoryOffset: linearMemoryOffset
                )
            )
        }

        mutating set(valuesNDArray) {
            var valuesNDArray = valuesNDArray

            let allAreUnmodifiedlDimensions = arrayShape
                .dimensions.lazy
                .map { $0 is UnmodifiedDimension }
                .reduce(true) { $0 && $1 }

            if !isKnownUniquelyReferenced(&data) || !allAreUnmodifiedlDimensions {
                let cp = copy()
                data = cp.data
                arrayShape = cp.arrayShape
            }

            var viewNDArray = self[ranges]
            let nElements = viewNDArray.shape.product()

            if viewNDArray.shape != valuesNDArray.shape {
                (viewNDArray, valuesNDArray) = broadcast(viewNDArray, and: valuesNDArray)
            }

            viewNDArray.data.value.withUnsafeMutableBufferPointer { view in
                valuesNDArray.data.value.withUnsafeBufferPointer { values in
                    for index in indexSequence(range: 0 ..< nElements, shape: viewNDArray.shape) {
                        let viewIndex = viewNDArray.linearIndex(at: index.rectangularIndex.value)
                        let valuesIndex = valuesNDArray.linearIndex(at: index.rectangularIndex.value)

                        view[viewIndex] = values[valuesIndex]
                    }
                }
            }
        }
    }

    @inlinable
    public subscript(_ ranges: ArrayRangeExpression...) -> NDArray {
        get {
            self[ranges]
        }
        set(value) {
            self[ranges] = value
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