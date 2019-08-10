import Foundation

public let all = ArrayRange.all
public let newAxis = ArrayRange.newAxis
public let ellipsis = ArrayRange.ellipsis
public let squeezeAxis = ArrayRange.squeezeAxis

// experimental alias
public let rest = ArrayRange.ellipsis
public let new = ArrayRange.newAxis
public let squeeze = ArrayRange.squeezeAxis

extension NDArray {
    @inlinable
    public subscript(_ ranges: [ArrayExpression]) -> NDArray {
        get {
            self[r: ranges.map { $0.arrayRange }]
        }
        mutating set(ndArray) {
            self[r: ranges.map { $0.arrayRange }] = ndArray
        }
    }

    @inlinable
    public subscript(r ranges: [ArrayRange]) -> NDArray {
        get {
            var ranges = ranges

            let nEllipsis = ranges.filter(isEllipsis).count

            precondition(nEllipsis <= 1, "A maximum of 1 .ellipsis can be used, got \(ranges)")

            if nEllipsis == 1 {
                let ellipsisIndex = ranges.firstIndex(where: isEllipsis)!
                let nAll = 1 + shape.count - ranges.count

                ranges.remove(at: ellipsisIndex)

                for _ in 0 ..< nAll {
                    ranges.insert(.all, at: ellipsisIndex)
                }
            }

            print(ranges)

            precondition(shape.count >= ranges.count)

            var dimensions = arrayShape.dimensions
            var linearMemoryOffset = arrayShape.linearMemoryOffset
            var dimensionToBeRemoved = [Int]()
            var dimensionToBeAdded = [Int: DimensionProtocol]()

            for (i, range) in ranges.enumerated() {
                switch range {
                case let .index(index):
                    linearMemoryOffset += dimensions[i].strideValue(of: index)
                    dimensionToBeRemoved.append(i)

                case let .range(range, stride: stride):
                    if range.lowerBound == 0, range.upperBound == dimensions[i].length, stride == 1 {
                        continue
                    }

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        end: range.upperBound,
                        stride: stride
                    )

                case let .closedRange(range, stride: stride):
                    if range.lowerBound == 0, range.upperBound + 1 == dimensions[i].length, stride == 1 {
                        continue
                    }

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        end: range.upperBound + 1,
                        stride: stride
                    )

                case let .partialRangeUpTo(range, stride: stride):
                    if range.upperBound == dimensions[i].length, stride == 1 {
                        continue
                    }
                    dimensions[i] = dimensions[i].sliced(
                        start: 0,
                        end: range.upperBound,
                        stride: stride
                    )

                case let .partialRangeThrough(range, stride: stride):

                    if range.upperBound + 1 == dimensions[i].length, stride == 1 {
                        continue
                    }

                    dimensions[i] = dimensions[i].sliced(
                        start: 0,
                        end: range.upperBound + 1,
                        stride: stride
                    )

                case let .partialRangeFrom(range, stride: stride):

                    if range.lowerBound == 0, stride == 1 {
                        continue
                    }

                    dimensions[i] = dimensions[i].sliced(
                        start: range.lowerBound,
                        stride: stride
                    )
                case .all:
                    continue
                case .squeezeAxis:
                    precondition(
                        dimensions[i].length == 1,
                        "Cannot squeeze dimension \(i) of \(shape), expected 1 got \(shape[i])"
                    )

                    linearMemoryOffset += dimensions[i].strideValue(of: 0)
                    dimensionToBeRemoved.append(i)

                case .newAxis:
                    dimensionToBeAdded[i] = SingularDimension()

                case .ellipsis:
                    fatalError("Ellipsis should be expand as a series of .all expressions")
                }
            }

            // TODO: this implementation is not correct due the fact the the length of dimension is changing
            // A correct way to implement this would be to do the operations sorted by the index
            // from high to low.
            dimensions = dimensions
                .enumerated()
                .filter { i, d in !dimensionToBeRemoved.contains(i) }
                .map { i, d in d }

            for (i, dimension) in dimensionToBeAdded {
                dimensions.insert(dimension, at: i)
            }

            return NDArray(
                data,
                shape: ArrayShape(
                    dimensions,
                    linearMemoryOffset: linearMemoryOffset
                )
            )
        }

        mutating set(ndArray) {
            var ndArray = ndArray

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

            if viewNDArray.shape != ndArray.shape {
                (viewNDArray, ndArray) = broadcast(viewNDArray, and: ndArray)
            }

            viewNDArray.data.value.withUnsafeMutableBufferPointer { view in
                ndArray.data.value.withUnsafeBufferPointer { values in
                    for index in indexSequence(range: 0 ..< nElements, shape: viewNDArray.shape) {
                        let viewIndex = viewNDArray.linearIndex(at: index.rectangularIndex)
                        let valuesIndex = ndArray.linearIndex(at: index.rectangularIndex)

                        view[viewIndex] = values[valuesIndex]
                    }
                }
            }
        }
    }

    @inlinable
    public subscript(_ ranges: ArrayExpression...) -> NDArray {
        get {
            self[ranges]
        }
        set(value) {
            self[ranges] = value
        }
    }

    @inlinable
    public subscript(r ranges: ArrayRange...) -> NDArray {
        get {
            self[r: ranges]
        }
        set(value) {
            self[r: ranges] = value
        }
    }
}

public protocol ArrayExpression {
    @inlinable
    var arrayRange: ArrayRange { get }
}

public enum ArrayRange: ArrayExpression {
    case ellipsis
    case newAxis
    case squeezeAxis
    case all
    case index(Int)
    case range(Range<Int>, stride: Int)
    case closedRange(ClosedRange<Int>, stride: Int)
    case partialRangeFrom(PartialRangeFrom<Int>, stride: Int)
    case partialRangeUpTo(PartialRangeUpTo<Int>, stride: Int)
    case partialRangeThrough(PartialRangeThrough<Int>, stride: Int)

    public var arrayRange: ArrayRange { self }
}

public func isEllipsis(_ range: ArrayRange) -> Bool {
    switch range {
    case .ellipsis:
        return true
    default:
        return false
    }
}

extension Int: ArrayExpression {
    public var arrayRange: ArrayRange { .index(self) }
}

extension Range: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .range(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .range(self, stride: stride)
    }
}

extension ClosedRange: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .closedRange(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .closedRange(self, stride: stride)
    }
}

extension PartialRangeFrom: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeFrom(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeFrom(self, stride: stride)
    }
}

extension PartialRangeUpTo: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeUpTo(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeUpTo(self, stride: stride)
    }
}

extension PartialRangeThrough: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .partialRangeThrough(self, stride: 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .partialRangeThrough(self, stride: stride)
    }
}