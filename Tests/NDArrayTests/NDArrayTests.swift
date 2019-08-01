import Foundation
@testable import NDArray
import XCTest

func timeIt(repetitions: Int = 1, function: () -> Void) -> Double {
    let startTime = Date()
    for _ in 1 ... repetitions {
        function()
    }
    return -startTime.timeIntervalSinceNow / Double(repetitions)
}

final class NDArrayTests: XCTestCase {
    func testElementWiseApply() {
        let a = NDArray([1, 2, 3], shape: [3])
        let b = NDArray([1, 2, 3], shape: [3])

        let c = elementWise(a, b, apply: +)

        XCTAssert(c.data == [2, 4, 6])
    }

    func testElementWiseApply2D() {
        let a = NDArray(
            [
                1, 2, 3,
                4, 5, 6,
            ],
            shape: [2, 3]
        )
        let b = NDArray(
            [
                1, 2, 3,
                4, 5, 6,
            ],
            shape: [2, 3]
        )

        let c = elementWise(a, b, apply: +)

        XCTAssertEqual(c.data, a.data.map { $0 * 2 })
    }

    func testConstructor() {
        let a = NDArray<Int>(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        let b = NDArray<Int>(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        let c = a + b

        XCTAssertEqual(a.shape, [2, 3])
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(c.data, a.data.map { $0 * 2 })
    }

    func testElementWiseApply3D() {
        let a = NDArray(
            [
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12,
            ],
            shape: [2, 2, 3]
        )
        let b = NDArray(
            [
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12,
            ],
            shape: [2, 2, 3]
        )

        let c = elementWise(a, b, apply: +)

        XCTAssertEqual(c.data, a.data.map { $0 * 2 })
    }

    func testElementWiseApply3DConstructor() {
        let a = NDArray<Int>(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [7, 8, 9],
                    [10, 11, 12],
                ],
            ]
        )
        let b = NDArray<Int>(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [7, 8, 9],
                    [10, 11, 12],
                ],
            ]
        )

        let c = a + b
        // print(c)

        XCTAssertEqual(a.shape, [2, 2, 3])
        XCTAssertEqual(b.shape, [2, 2, 3])
        XCTAssertEqual(c.shape, [2, 2, 3])
        XCTAssertEqual(c.data, a.data.map { $0 * 2 })
    }

    func testElementWiseApplyParallel() {
        let a = NDArray(Array(1 ... 100), shape: [100])
        let b = NDArray(Array(1 ... 100), shape: [100])

        let c = elementWiseInParallel(a, b, apply: +)

        XCTAssert(c.data == a.data.map { $0 * 2 })
    }

    func testIndex() {
        let a = NDArray([
            3, 30,
            2, 20,
            1, 10,

        ], shape: [3, 2])

        let b = a[1, 1]

        let realIndex = b.realIndex(of: 0)

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(b.data[realIndex], 20)
    }

    func testScalarElementWiseAdd() {
        let a = NDArray([
            3, 30,
            2, 20,
            1, 10,

        ], shape: [3, 2])

        let b = a[1, 1]
        let c = a[2, 0]

        let d = b + c

        XCTAssertEqual(b.copy().data, [20])
        XCTAssertEqual(c.copy().data, [1])
        XCTAssertEqual(d.data, [21])
    }

    // func testElementWiseApplyParallelBenchmark() {
    //     let a = Array(1 ... 20_000_000)
    //     let b = Array(1 ... 20_000_000)

    //     let timeParallel = timeIt(repetitions: 1) {
    //         _ = elementWiseInParallel(a, b, apply: +)
    //     }
    //     let timeSerial = timeIt(repetitions: 1) {
    //         _ = elementWise(a, b, apply: +)
    //     }

    //     print("time parallel:", timeParallel)
    //     print("time serial:", timeSerial)
    // }

    func testTransposed() {
        let a = NDArray<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ]).transposed([1, 0]).copy()

        XCTAssertEqual(a.data, [1, 4, 2, 5, 3, 6])
    }

    func testRangesSplit() {
        let ranges = splitRanges(total: 70, splits: 11)

        XCTAssert(ranges.count == 11)
    }

    func testAnything() {
        print(NDArray(scalar: 0))
        print(NDArray([1, 2, 3, 4]))
        print(NDArray<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ]))
        print(NDArray<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ]).transposed([1, 0]))
        print(NDArray<Int>(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [7, 8, 9],
                    [10, 11, 12],
                ],
            ]
        ))

        print(NDArray<Int>([
            [1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10],
        ])[0..., 1 ..< 5][0..., ..<3])

        print(NDArray<Int>([
            [1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8, 9, 10],
        ])[0..., ((-1)...).stride(2)])

        print(NDArray([1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10])[((-1)...).stride(-1)])
    }

    static var allTests = [
        ("testElementWiseApply", testElementWiseApply),
        ("testElementWiseApply2D", testElementWiseApply2D),
        ("testElementWiseApply3D", testElementWiseApply3D),
        ("testElementWiseApplyParallel", testElementWiseApplyParallel),
        // ("testElementWiseApplyParallelBenchmark", testElementWiseApplyParallelBenchmark),
        ("testRangesSplit", testRangesSplit),
        ("testIndex", testIndex),
        ("testScalarElementWiseAdd", testScalarElementWiseAdd),
        ("testConstructor", testConstructor),
        ("testElementWiseApply3DConstructor", testElementWiseApply3DConstructor),
        ("testAnything", testAnything),
        ("testTransposed", testTransposed),
    ]
}