
protocol ShapeProtocol {
    // func indexIterator() -> AnyIterator<Int>
}

public struct Shape: ShapeProtocol {
    var dimensions: [DimensionProtocol]

    public init(_ shape: [Int]) {
        let N = shape.count

        var acc = 1
        var memory_strides = [Int]()

        for i in 0 ..< shape.count {
            if shape[N - i - 1] > 1 {
                memory_strides.append(acc)
                acc *= shape[N - i - 1]
            } else {
                memory_strides.append(0)
            }
        }

        dimensions = memory_strides
            .reversed()
            .enumerated()
            .map { i, memory_stride -> Dimension in
                Dimension(
                    memory_stride: memory_stride, total: shape[i],
                    start: 0, end: shape[i],
                    stride: 1, repetitions: 1
                )
            }
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