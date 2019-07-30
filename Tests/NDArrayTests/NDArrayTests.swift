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
        let a = [1, 2, 3]
        let b = [1, 2, 3]

        let c = elementWise(a, b, apply: +)

        XCTAssert(c == [2, 4, 6])
    }

    func testElementWiseApplyParallel() {
        let a = Array(1 ... 100)
        let b = Array(1 ... 100)

        let c = elementWiseInParallel(a, b, apply: +)

        XCTAssert(c == a.map { $0 * 2 })
    }

    func testElementWiseApplyParallelBenchmark() {
        let a = Array(1 ... 20_000_000)
        let b = Array(1 ... 20_000_000)

        let timeParallel = timeIt(repetitions: 1) {
            _ = elementWiseInParallel(a, b, apply: +)
        }
        let timeSerial = timeIt(repetitions: 1) {
            _ = elementWise(a, b, apply: +)
        }

        print("time parallel:", timeParallel)
        print("time serial:", timeSerial)
    }

    func testRangesSplit() {
        let ranges = splitRanges(total: 70, splits: 11)

        XCTAssert(ranges.count == 11)
    }

    static var allTests = [
        ("testElementWiseApply", testElementWiseApply),
        ("testElementWiseApplyParallel", testElementWiseApplyParallel),
        ("testElementWiseApplyParallelBenchmark", testElementWiseApplyParallelBenchmark),
        ("testRangesSplit", testRangesSplit),
    ]
}