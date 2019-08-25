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

struct Point: AdditiveArithmetic {
    let x: Float
    let y: Float

    static var zero: Point { Point(x: 0, y: 0) }

    static prefix func + (lhs: Self) -> Self {
        lhs
    }

    static func + (lhs: Self, rhs: Self) -> Self {
        Point(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }

    static func += (lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }

    static func - (lhs: Self, rhs: Self) -> Self {
        Point(x: lhs.x - rhs.x, y: lhs.y - rhs.y)
    }

    public static func -= (_ lhs: inout Point, _ rhs: Point) {
        lhs = lhs - rhs
    }
}

final class NDArrayTests: XCTestCase {
    func testElementWiseApply() {
        let a = NDArray<Int>([1, 2, 3], shape: [3])
        let b = NDArray<Int>([1, 2, 3], shape: [3])

        let c = a + b

        XCTAssert(c.toArray([Int].self) == [2, 4, 6])
    }

    func testElementWiseApply2D() {
        let a = NDArray<Int>(
            [
                1, 2, 3,
                4, 5, 6,
            ],
            shape: [2, 3]
        )
        let b = NDArray<Int>(
            [
                1, 2, 3,
                4, 5, 6,
            ],
            shape: [2, 3]
        )

        let c = a + b

        XCTAssertEqual(
            c.toArray([[Int]].self),
            a.toArray([[Int]].self).map { arr in
                arr.map { x in x * 2 }
            }
        )
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
        XCTAssertEqual(
            c.toArray([[Int]].self),
            a.toArray([[Int]].self).map { arr in
                arr.map { x in x * 2 }
            }
        )
    }

    func testElementWiseApply3D() {
        let a = NDArray<Int>(
            [
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12,
            ],
            shape: [2, 2, 3]
        )
        let b = NDArray<Int>(
            [
                1, 2, 3,
                4, 5, 6,

                7, 8, 9,
                10, 11, 12,
            ],
            shape: [2, 2, 3]
        )

        let c = a + b

        XCTAssertEqual(
            c.toArray([[[Int]]].self),
            a.toArray([[[Int]]].self).map { d0 in
                d0.map { d1 in
                    d1.map { d2 in d2 * 2 }
                }
            }
        )
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

        XCTAssertEqual(a.shape, [2, 2, 3])
        XCTAssertEqual(b.shape, [2, 2, 3])
        XCTAssertEqual(c.shape, [2, 2, 3])
        XCTAssertEqual(
            c.toArray([[[Int]]].self),
            a.toArray([[[Int]]].self).map { d0 in
                d0.map { d1 in
                    d1.map { d2 in d2 * 2 }
                }
            }
        )
    }

    func testExample() {
        let a = NDArray<Int>(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
        let b = NDArray<Int>(
            [
                [7, 8, 9],
                [10, 11, 12],
            ]
        )

        _ = (a + b) * a
    }

    func testCustomType() {
        let a = NDArray<Point>(
            [Point(x: 1, y: 2), Point(x: 2, y: 3)]
        )
        let b = NDArray<Point>(
            [Point(x: 4, y: 5), Point(x: 6, y: 7)]
        )

        let c = a + b

        let target = NDArray<Point>(
            [Point(x: 5, y: 7), Point(x: 8, y: 10)]
        )

        XCTAssertEqual(
            c.toArray([Point].self),
            target.toArray([Point].self)
        )
    }

    func testCustomType2() {
        let a = NDArray<Point>(
            [Point(x: 1, y: 2), Point(x: 2, y: 3)]
        )
        let b = NDArray<Point>(
            [Point(x: 4, y: 5), Point(x: 6, y: 7)]
        )

        let c: NDArray<Point> = elementwise(a, b, apply: +)

        let target = NDArray<Point>(
            [Point(x: 5, y: 7), Point(x: 8, y: 10)]
        )

        XCTAssertEqual(
            c.toArray([Point].self),
            target.toArray([Point].self)
        )
    }

    func testElementWiseApplyParallel() {
        let a = NDArray<Int>(Array(1 ... 100), shape: [100])
        let b = NDArray<Int>(Array(1 ... 100), shape: [100])

        let c = a + b

        XCTAssert(
            c.toArray([Int].self) == a.toArray([Int].self).map { $0 * 2 }
        )
    }

    func testIndex() {
        let a = NDArray<Int>([
            3, 30,
            2, 20,
            1, 10,

        ], shape: [3, 2])

        let b = a[1, 1]

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 20)
    }

    func testScalarElementWiseAdd() {
        let a = NDArray<Int>([
            3, 30,
            2, 20,
            1, 10,

        ], shape: [3, 2])

        let b = a[1, 1]
        let c = a[2, 0]

        let d = b + c

        XCTAssertEqual(b.baseCopy().data.value, [20])
        XCTAssertEqual(c.baseCopy().data.value, [1])
        XCTAssertEqual(d.toArray([Int].self), [21])
    }

    func testBroadcast1() {
        let a = NDArray<Int>([1, 2, 3, 4], shape: [1, 4])
        let b = NDArray<Int>([1, 2, 3, 4], shape: [4, 1])

        let c = a + b

        let target = NDArray<Int>([
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ])

        XCTAssertEqual(c.baseCopy().data.value, target.baseCopy().data.value)
    }

    func testBroadcast2() {
        let a = NDArray<Int>([1, 2, 3, 4], shape: [1, 4])
        let b = 1

        let c = a + b

        let target = NDArray<Int>([
            2, 3, 4, 5,
        ])

        XCTAssertEqual(c.baseCopy().data.value, target.baseCopy().data.value)
    }

    func testBroadcast3() {
        let a = NDArray<Int>([1, 2, 3, 4], shape: [1, 4])
        let b = 2

        let c = a * b

        let target = NDArray<Int>([
            2, 4, 6, 8,
        ])

        XCTAssertEqual(c.baseCopy().data.value, target.baseCopy().data.value)
    }

    func testAssign() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0..] = NDArray<Int>([1, 1, 1, 1])
        b[0..] = NDArray<Int>([2, 2, 2, 2])

        XCTAssertEqual(a.baseCopy().data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 2, 2])
    }

    func testAssign2() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0..] = NDArray(1)
        b[0..] = NDArray(2)

        XCTAssertEqual(a.baseCopy().data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 2, 2])
    }

    func testAssign3() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0..] = NDArray(1)
        b[0..] = NDArray(2)

        XCTAssertEqual(a.baseCopy().data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 2, 2])
    }

    func testAssign4() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0..] = [1, 1, 1, 1]
        b[0..] = [2, 2, 2, 2]

        XCTAssertEqual(a.baseCopy().data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 2, 2])
    }

    func testAssign5() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a
        var c = b[1..-1]

        a[0..] = [1, 1, 1, 1]
        b[0..] = [2, 2, 2, 2]

        let d = c[0..]

        c[0..] = NDArray(10)

        XCTAssertEqual(a.baseCopy().data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 2, 2])
        XCTAssertEqual(c.baseCopy().data.value, [10, 10])
        XCTAssertEqual(d.baseCopy().data.value, [2, 3])
    }

    func testTransposed() {
        let a = NDArray<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ]).transposed([1, 0])

        XCTAssertEqual(a.baseCopy().data.value, [1, 4, 2, 5, 3, 6])
    }

    func testRangesSplit() {
        let ranges = splitRanges(total: 70, splits: 11)

        XCTAssert(ranges.count == 11)
    }

    func testSqueezeAxis() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
        ])

        let b = a[squeeze, all]

        XCTAssertEqual(a.shape, [1, 4])
        XCTAssertEqual(b.shape, [4])
    }

    func testNewAxis() {
        let a = NDArray<Int>(
            [1, 2, 3, 4]
        )
        let b = a[new]

        XCTAssertEqual(a.shape, [4])
        XCTAssertEqual(b.shape, [1, 4])
    }

    func testAll() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ])
        let b = a[ArrayRange.all, 1]

        XCTAssertEqual(b.baseCopy().data.value, [2, 3])
    }

    func testEllipsis() {
        let a = NDArray<Int>(
            Array(1 ... 16),
            shape: [2, 2, 2, 2]
        )
        let b = a[ArrayRange.ellipsis]

        XCTAssertEqual(b.shape, [2, 2, 2, 2])
    }

    func testEllipsis2() {
        let a = NDArray<Int>(
            Array(1 ... 16),
            shape: [2, 2, 2, 2]
        )
        let b = a[0, ArrayRange.ellipsis]

        XCTAssertEqual(b.shape, [2, 2, 2])
    }

    func testEllipsis3() {
        let a = NDArray<Int>(
            Array(1 ... 16),
            shape: [2, 2, 2, 2]
        )
        let b = a[0, rest, 0]

        XCTAssertEqual(b.shape, [2, 2])
    }

    func testEllipsis4() {
        let a = NDArray<Int>(
            Array(1 ... 16),
            shape: [2, 2, 2, 2]
        )
        let b = a[0, ArrayRange.ellipsis, 0, 0]

        XCTAssertEqual(b.shape, [2])
    }

    func testEllipsis5() {
        let a = NDArray<Int>(
            Array(1 ... 16),
            shape: [2, 2, 2, 2]
        )
        let b = a[0, 0, rest, 0, 0]

        XCTAssertEqual(b.shape, [])
    }

    func testNegativeStride() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[((-1)...).stride(-1)]

        XCTAssertEqual(a.baseCopy().data.value, b.baseCopy().data.value.reversed())
    }

    func testNegativeStride2() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[....-1]

        XCTAssertEqual(a.baseCopy().data.value, b.baseCopy().data.value.reversed())
    }

    func testNegativeStride3() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[..1..-2]

        XCTAssertEqual(b.baseCopy().data.value, [5, 3])
    }

    func testNegativeIndex() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[-1]

        XCTAssertEqual(b.baseCopy().data.value, [5])
    }

    func testFilter() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[[1, 1, 0, 2, -1]]

        XCTAssertEqual(b.baseCopy().data.value, [2, 2, 1, 3, 5])
    }

    func testFilter2() {
        let a = NDArray<Int>([1, 2, 3, 4, 5])

        let b = a[[true, true, false, false, true]]

        XCTAssertEqual(b.baseCopy().data.value, [1, 2, 5])
    }

    func testReduce1() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.reduce(
            axis: [0],
            initial: NDArray<Int>([0, 0, 0, 0]),
            f: +
        )

        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.baseCopy().data.value, [11, 22, 33, 44])
    }

    func testReduce2() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.reduce(
            axis: [1],
            initial: NDArray<Int>([0, 0]),
            f: +
        )

        XCTAssertEqual(b.shape, [2])
        XCTAssertEqual(b.baseCopy().data.value, [10, 100])
    }

    func testReduce3() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.reduce(
            axis: [0, 1],
            initial: NDArray<Int>(0),
            f: +
        )

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 110)
    }

    func testSum1() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.sum(
            axis: [0]
        )

        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.baseCopy().data.value, [11, 22, 33, 44])
    }

    func testSum2() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.sum(
            axis: [1]
        )

        XCTAssertEqual(b.shape, [2])
        XCTAssertEqual(b.baseCopy().data.value, [10, 100])
    }

    func testSum3() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.sum(
            axis: [0, 1]
        )

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 110)
    }

    func testMean1() {
        let a: NDArray<Float> = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ]

        let b = a.mean(
            axis: [0]
        )

        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.baseCopy().data.value, [11 / 2, 22 / 2, 33 / 2, 44 / 2] as [Float])
    }

    func testMean2() {
        let a: NDArray<Float> = [
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ]

        let b = a.mean(
            axis: [1]
        )

        XCTAssertEqual(b.shape, [2])
        XCTAssertEqual(b.baseCopy().data.value, [10 / 4, 100 / 4] as [Float])
    }

    func testMean3() {
        let a = NDArray<Float>([[Float]]([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ]))

        let b = a.mean(
            axis: [0, 1]
        )

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 110 / 8)
    }

    func testMax1() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.max(
            axis: [0]
        )

        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.baseCopy().data.value, [10, 20, 30, 40])
    }

    func testMax2() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.max(
            axis: [1]
        )

        XCTAssertEqual(b.shape, [2])
        XCTAssertEqual(b.baseCopy().data.value, [4, 40])
    }

    func testMax3() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.max(
            axis: [0, 1]
        )

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 40)
    }

    func testMin1() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.min(
            axis: [0]
        )

        XCTAssertEqual(b.shape, [4])
        XCTAssertEqual(b.baseCopy().data.value, [1, 2, 3, 4])
    }

    func testMin2() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.min(
            axis: [1]
        )

        XCTAssertEqual(b.shape, [2])
        XCTAssertEqual(b.baseCopy().data.value, [1, 10])
    }

    func testMin3() {
        let a = NDArray<Int>([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
        ])

        let b = a.min(
            axis: [0, 1]
        )

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(b.scalarized(), 1)
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
        ("testTransposed", testTransposed),
        ("testExample", testExample),
        ("testCustomType", testCustomType),
        ("testCustomType2", testCustomType2),
        ("testBroadcast1", testBroadcast1),
        ("testBroadcast2", testBroadcast2),
        ("testBroadcast3", testBroadcast3),
        ("testAssign", testAssign),
        ("testAssign2", testAssign2),
        ("testAssign3", testAssign3),
        ("testAssign4", testAssign4),
        ("testAssign5", testAssign5),
        ("testSqueezeAxis", testSqueezeAxis),
        ("testNewAxis", testNewAxis),
        ("testAll", testAll),
        ("testEllipsis", testEllipsis),
        ("testEllipsis2", testEllipsis2),
        ("testEllipsis3", testEllipsis3),
        ("testEllipsis4", testEllipsis4),
        ("testEllipsis5", testEllipsis5),
        ("testNegativeStride", testNegativeStride),
        ("testNegativeStride2", testNegativeStride2),
        ("testNegativeStride3", testNegativeStride3),
        ("testNegativeIndex", testNegativeIndex),
        ("testFilter", testFilter),
        ("testFilter2", testFilter2),
        ("testReduce1", testReduce1),
        ("testReduce2", testReduce2),
        ("testReduce3", testReduce3),
        ("testSum1", testSum1),
        ("testSum2", testSum2),
        ("testSum3", testSum3),
        ("testMean1", testMean1),
        ("testMean2", testMean2),
        ("testMean3", testMean3),
        ("testMax1", testMax1),
        ("testMax2", testMax2),
        ("testMax3", testMax3),
        ("testMin1", testMin1),
        ("testMin2", testMin2),
        ("testMin3", testMin3),
    ]
}