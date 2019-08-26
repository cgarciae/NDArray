

public struct ArrayShape {
    @usableFromInline let dimensions: [DimensionProtocol]
    @usableFromInline let shape: [Int]

    @inlinable
    public init(_ shape: [Int]) {
        self.shape = shape

        dimensions = (0 ..< shape.count)
            .map { i -> DimensionProtocol in

                if shape[i] == 1 {
                    return SingularDimension()
                } else {
                    return Dimension(length: shape[i])
                }
            }
    }

    @usableFromInline
    init(_ dimensions: [DimensionProtocol]) {
        self.dimensions = dimensions
        shape = dimensions.map { $0.length }
    }

    // @inlinable
    // public func linearIndex(of indexes: UnsafeMutableBufferPointer<Int>) -> Int {
    //     let partialIndex = zip(indexes, dimensions)
    //         .lazy
    //         .map { index, dimension in
    //             dimension.strideValue(of: index)
    //         }
    //         .sum()

    //     return partialIndex + linearMemoryOffset
    // }

    public subscript(_ ranges: [ArrayRange]) -> ArrayShape {
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

        precondition(shape.count >= ranges.count)

        var dimensions = self.dimensions

        for (i, range) in ranges.enumerated().reversed() {
            switch range {
            case .index:
                dimensions.remove(at: i)

            case let .slice(start: start, end: end, stride: stride):

                if start == 0, end! == dimensions[i].length, stride == 1 {
                    continue
                }

                dimensions[i] = dimensions[i].sliced(
                    start: start!,
                    end: end!,
                    stride: stride
                )
            case let .filter(indexes):
                dimensions[i] = dimensions[i].select(indexes: indexes)

            case .all:
                continue
            case .squeezeAxis:
                precondition(
                    dimensions[i].length == 1,
                    "Cannot squeeze dimension \(i) of \(shape), expected 1 got \(shape[i])"
                )

                dimensions.remove(at: i)

            case .newAxis:
                dimensions.insert(SingularDimension(), at: i)

            case .ellipsis:
                fatalError("Ellipsis should be expand as a series of .all expressions")
            }
        }

        return ArrayShape(dimensions)
    }
}