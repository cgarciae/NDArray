import Foundation

// print(Array(ContiguousDimension(0 ..< 10)))
print(Array(Dimension(start: 0, end: 10, stride: 1, repetitions: 1).indexIterator()))
print(Dimension(start: 0, end: 10, stride: 1, repetitions: 1).count)
print(Dimension(start: 0, end: 10, stride: 1, repetitions: 1).realCount)
print(Dimension(start: 0, end: 10, stride: 1, repetitions: 1).realIndex(of: 2))
print(Array(Dimension(start: 0, end: 10, stride: 3, repetitions: 1).indexIterator()))
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 1).count)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 1).realCount)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 1).realIndex(of: 2))
print(Array(Dimension(start: 0, end: 10, stride: 3, repetitions: 2).indexIterator()))
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 2).count)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 2).realCount)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 2).realIndex(of: 2))
print(Array(Dimension(start: 0, end: 10, stride: 3, repetitions: 3).indexIterator()))
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 3).count)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 3).realCount)
print(Dimension(start: 0, end: 10, stride: 3, repetitions: 3).realIndex(of: 2))
// print(Array(Dimension(start: 0, end: 10, stride: 3, repetitions: 1)
//         .fullSequence(cycleRepetitions: 2, elementRepetitions: 3)))
// print(Dimension(start: 0, end: 10, stride: 3, repetitions: 1).count)
// print(Array(Dimension(start: 0, end: 10, stride: 3, repetitions: 1)
//         .fullSequence(cycleRepetitions: 2, elementRepetitions: 3))[3])

let shape = Shape(dimensions: [
    Dimension(start: 0, end: 2, stride: 1, repetitions: 1),
    Dimension(start: 0, end: 2, stride: 1, repetitions: 1),
    Dimension(start: 0, end: 2, stride: 1, repetitions: 2),
])

print(Array(shape.dimensions[0].indexIterator()), shape.dimensions[0].count)
print(Array(shape.dimensions[1].indexIterator()), shape.dimensions[1].count)

// print(Array(shape.dimensions[0].fullSequence(cycleRepetitions: 1, elementRepetitions: 4).makeIterator()))
// print(Array(shape.dimensions[1].fullSequence(cycleRepetitions: 4, elementRepetitions: 1).makeIterator()))

print(Array(shape.indexIterator()))
"""
[ 
    0, 1, 
    0, 1, 
    2, 3, 
    2, 3, 
    
    0, 1, 
    0, 1, 
    2, 3, 
    2, 3
]
"""

func timeIt(repetitions: Int = 1, function: () -> Void) -> Double {
    let startTime = Date()
    for _ in 1 ... repetitions {
        function()
    }
    return -startTime.timeIntervalSinceNow / Double(repetitions)
}

// let a = Array(1 ... 200_000)
// let b = Array(1 ... 200_000)

// let timeParallel = timeIt(repetitions: 100) {
//     _ = elementWiseInParallel(a, b) { (a: Int, b: Int) -> Float in
//         cos(Float((a * b + a) * b))
//     }
// }

// let timeSerial = timeIt(repetitions: 100) {
//     _ = elementWise(a, b) { (a: Int, b: Int) -> Float in
//         cos(Float((a * b + a) * b))
//     }
// }

// print("time parallel:", timeParallel)
// print("time serial:", timeSerial)