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

        XCTAssert(c.data.value == [2, 4, 6])
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

        XCTAssertEqual(c.data.value, a.data.value.map { $0 * 2 })
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
        XCTAssertEqual(c.data.value, a.data.value.map { $0 * 2 })
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

        XCTAssertEqual(c.data.value, a.data.value.map { $0 * 2 })
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
        XCTAssertEqual(c.data.value, a.data.value.map { $0 * 2 })
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

    func testExample2() {
        let a = NDArray<Point>(
            [Point(x: 1, y: 2), Point(x: 2, y: 3)]
        )
        let b = NDArray<Point>(
            [Point(x: 4, y: 5), Point(x: 6, y: 7)]
        )

        _ = a + b
    }

    func testElementWiseApplyParallel() {
        let a = NDArray<Int>(Array(1 ... 100), shape: [100])
        let b = NDArray<Int>(Array(1 ... 100), shape: [100])

        let c = a + b

        XCTAssert(c.data.value == a.data.value.map { $0 * 2 })
    }

    func testIndex() {
        let a = NDArray<Int>([
            3, 30,
            2, 20,
            1, 10,

        ], shape: [3, 2])

        let b = a[1, 1]

        let realIndex = b.realIndex(of: 0)

        XCTAssertEqual(b.shape, [])
        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(b.data.value[realIndex], 20)
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

        XCTAssertEqual(b.copy().data.value, [20])
        XCTAssertEqual(c.copy().data.value, [1])
        XCTAssertEqual(d.data.value, [21])
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

        XCTAssertEqual(c.data.value, target.data.value)
    }

    func testBroadcast2() {
        let a = NDArray<Int>([1, 2, 3, 4], shape: [1, 4])
        let b = 1

        let c = a + b

        let target = NDArray<Int>([
            2, 3, 4, 5,
        ])

        XCTAssertEqual(c.data.value, target.data.value)
    }

    func testBroadcast3() {
        let a = NDArray<Int>([1, 2, 3, 4], shape: [1, 4])
        let b = 2

        let c = a * b

        let target = NDArray<Int>([
            2, 4, 6, 8,
        ])

        XCTAssertEqual(c.data.value, target.data.value)
    }

    func testAssign() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0...] = NDArray<Int>([1, 1, 1, 1])
        b[0...] = NDArray<Int>([2, 2, 2, 2])

        XCTAssertEqual(a.data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.data.value, [2, 2, 2, 2])
    }

    func testAssign2() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0...] = NDArray(1)
        b[0...] = NDArray(2)

        XCTAssertEqual(a.data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.data.value, [2, 2, 2, 2])
    }

    func testAssign3() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0...] = NDArray(1)
        b[0...] = NDArray(2)

        XCTAssertEqual(a.data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.data.value, [2, 2, 2, 2])
    }

    func testAssign4() {
        var a = NDArray<Int>([1, 2, 3, 4], shape: [4])
        var b = a

        a[0...] = [1, 1, 1, 1]
        b[0...] = [2, 2, 2, 2]

        XCTAssertEqual(a.data.value, [1, 1, 1, 1])
        XCTAssertEqual(b.data.value, [2, 2, 2, 2])
    }

    func testTransposed() {
        let a = NDArray<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ]).transposed([1, 0]).copy()

        XCTAssertEqual(a.data.value, [1, 4, 2, 5, 3, 6])
    }

    func testRangesSplit() {
        let ranges = splitRanges(total: 70, splits: 11)

        XCTAssert(ranges.count == 11)
    }

    func testDifferentiable() {
        let a: NDArray<Float> = [1, 2, 3, 4]

        let da = a.gradient { a -> Float in
            let x = a.sum()

            return x * x
        }

        print(da)
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
        ("testExample2", testExample2),
        ("testBroadcast1", testBroadcast1),
        ("testBroadcast2", testBroadcast2),
        ("testBroadcast3", testBroadcast3),
        ("testAssign", testAssign),
        ("testAssign2", testAssign2),
        ("testAssign3", testAssign3),
        ("testAssign4", testAssign4),
        ("testDifferentiable", testDifferentiable),
    ]
}