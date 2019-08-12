import NDArray

let a = NDArray<Int>([1, 2, 3, 4, 5])

let b = a[((-1)...).stride(-1)]

// let a = NDArray<Int>([
//     [1, 2, 3],
//     [4, 5, 6],
// ]).transposed([1, 0]).copy()

// let s = indexSequence(range: 3 ..< 10, shape: [10, 2, 5])

// for x in s {
//     print(x.rectangularIndex)
// }

// print(NDArray<Int>(0))
// print(NDArray<Int>([1, 2, 3, 4]))
// print(NDArray<Int>([
//     [1, 2, 3],
//     [4, 5, 6],
// ]))
// print(NDArray<Int>([
//     [1, 2, 3],
//     [4, 5, 6],
// ]).transposed([1, 0]))
// print(NDArray<Int>(
//     [
//         [
//             [1, 2, 3],
//             [4, 5, 6],
//         ],
//         [
//             [7, 8, 9],
//             [10, 11, 12],
//         ],
//     ]
// ))

// print(NDArray<Int>([
//     [1, 2, 3, 4, 5, 6, 7],
//     [4, 5, 6, 7, 8, 9, 10],
// ])[0..., 1 ..< 5][0..., ..<3])

// print(NDArray<Int>([
//     [1, 2, 3, 4, 5, 6, 7],
//     [4, 5, 6, 7, 8, 9, 10],
// ])[0..., ((-1)...).stride(2)])