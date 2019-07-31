
public protocol ShapeProtocol {
    // func indexIterator() -> AnyIterator<Int>
}

public struct Shape: ShapeProtocol {
    public let dimensions: [DimensionProtocol]
    public let dimension_lengths: [Int]

    public init(_ shape: [Int]) {
        let memory_strides = shape.reversed().scan(*).reversed()

        dimensions = (0 ..< shape.count)
            .map { i -> DimensionProtocol in
                let length = shape[i]

                if length == 1 {
                    return SingularDimension()
                } else if i == shape.count - 1 {
                    return Dimension(length: length, memory_stride: 1)
                } else {
                    return Dimension(length: length, memory_stride: memory_strides[i + 1])
                }
            }

        dimension_lengths = dimensions.map { $0.length }
    }

    public init(_ dimensions: [DimensionProtocol]) {
        self.dimensions = dimensions
        dimension_lengths = dimensions.map { $0.length }
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        if index == 0 { return 0 }

        return dimensions
            .lazy
            .filter { $0.memory_layout.stride > 0 }
            .reduce((accumulatedIndex: 0, reminder: index)) { state, dimension in

                if state.reminder == 0 {
                    return state
                }

                let (quotient, reminder) = state.reminder.quotientAndRemainder(dividingBy: dimension.memory_layout.stride)
                let accumulatedIndex = dimension.memoryStridedValue(of: quotient) + state.accumulatedIndex

                return (accumulatedIndex: accumulatedIndex, reminder: reminder)
            }
            .accumulatedIndex
    }

    @inlinable
    public func realIndex(of indexes: [Int]) -> Int {
        precondition(
            indexes.count == dimension_lengths.count,
            "Invalid index dimensions, expected \(dimension_lengths.count), got \(indexes.count)"
        )
        precondition(
            zip(indexes, dimension_lengths).map(<=).reduce(true) { $0 && $1 },
            "Index out of bounds, expected values in the range of \(dimension_lengths), got \(indexes)"
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

// extension Shape {

//     func indexIterator() -> AnyIterator<Int> {
//         let sequence = AnySequence { () -> AnyIterator<Int> in

//             let dimensionCounts = self.dimensions.map { $0.count }
//             var cumulativeProd = Array(dimensionCounts
//                 .reversed()
//                 .reduce([]) { (acc: [Int], x: Int) -> [Int] in
//                     var acc = acc
//                     if acc.count == 0 {
//                         return [x]
//                     } else {
//                         acc.append(acc.last! * x)
//                         return acc
//                     }
//                 }
//                 .reversed()
//             )

//             var reps = Array(cumulativeProd)
//             reps = Array(reps[1...])
//             reps.append(1)
//             // reps.reverse()

//             // let totalProd = cumulativeProd.reduce(1, +)

//             let N = dimensionCounts.count

//             print(cumulativeProd)
//             print(reps)

//             let dimensionIndexIterators = (0 ..< cumulativeProd.count).map { i -> AnyIterator<Int> in
//                 let cycleRepetitions = cumulativeProd[N - i - 1] / dimensionCounts[i]
//                 let elementRepetitions = reps[i]
//                 let dim = self.dimensions[i]

//                 return dim
//                     .fullSequence(cycleRepetitions: cycleRepetitions, elementRepetitions: elementRepetitions)
//                     .makeIterator()
//             }

//             return AnyIterator { () -> Int? in
//                 let res = zip(dimensionIndexIterators, reps)
//                     .map { iterator, prod -> Int? in
//                         if let next = iterator.next() {
//                             return next * prod
//                         } else {
//                             return nil
//                         }
//                     }
//                     .reduce(0) { (acc: Int?, x: Int?) -> Int? in
//                         guard let acc = acc, let x = x else { return nil }

//                         return acc + x
//                     }

//                 return res
//             }
//         }

//         return sequence.makeIterator()
//     }
// }