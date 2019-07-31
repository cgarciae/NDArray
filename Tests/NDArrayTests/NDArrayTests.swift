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

        print(b.shape)

        let realIndex = b._shape.realIndex(of: [0, 0])

        XCTAssertEqual(b.data[realIndex], 20)
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

    func testRangesSplit() {
        let ranges = splitRanges(total: 70, splits: 11)

        XCTAssert(ranges.count == 11)
    }

    static var allTests = [
        ("testElementWiseApply", testElementWiseApply),
        ("testElementWiseApplyParallel", testElementWiseApplyParallel),
        // ("testElementWiseApplyParallelBenchmark", testElementWiseApplyParallelBenchmark),
        ("testRangesSplit", testRangesSplit),
        ("testIndex", testIndex),
    ]
}