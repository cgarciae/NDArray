// import Foundation
@testable import NDArray
import XCTest

final class DimensionTests: XCTestCase {
    func testVirualEmpty() {
        let dimension = Dimension(memory_stride: 0, total: 0, repetitions: 3)

        // let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        // XCTAssertEqual(realIndex, 0)
        XCTAssertEqual(count, 0)
        XCTAssertEqual(realCount, 0)
    }

    func testVirualOneRepeated() {
        let dimension = Dimension(memory_stride: 0, total: 1, repetitions: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 0)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 1)
    }

    func testVirualStrided() {
        let dimension = Dimension(memory_stride: 0, total: 10, stride: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 4)
        XCTAssertEqual(realCount, 10)
    }

    func testVirualStridedMultiple() {
        let dimension = Dimension(memory_stride: 0, total: 4, stride: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 2)
        XCTAssertEqual(realCount, 4)
    }

    func testVirualStridedMultiple2() {
        let dimension = Dimension(memory_stride: 0, total: 7, stride: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 7)
    }

    func testVirualStridedMultiple3() {
        let dimension = Dimension(memory_stride: 0, total: 8, stride: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 8)
    }

    func testVirualStridedMultiple4() {
        let dimension = Dimension(memory_stride: 0, total: 9, stride: 3)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 9)
    }

    func testVirualStridedMultiple5() {
        let dimension = Dimension(memory_stride: 0, total: 9, start: 5, end: 6, stride: 3)

        let realIndex = dimension.realIndex(of: 0)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 5)
        XCTAssertEqual(count, 1)
        XCTAssertEqual(realCount, 9)
    }

    func testVirualEmpty2() {
        let dimension = Dimension(memory_stride: 0, total: 1, repetitions: 3)

        let realIndex = dimension.realIndex(of: 1)

        XCTAssertEqual(realIndex, 0)
    }

    func testScan() {
        let accumulatedProduct = [1, 2, 3, 4].scan(*)

        let total = accumulatedProduct.reduce(0) { (acc: Int, x: Int) -> Int in acc + x }

        XCTAssertEqual(total, 33)
    }

    func testShapeInit() {
        let shape = Shape([5, 1, 4, 2])

        XCTAssertEqual(
            shape.dimensions.map { $0.memory_stride },
            [8, 0, 2, 1]
        )
    }

    static var allTests = [
        ("testVirualEmpty", testVirualEmpty),
        ("testVirualOneRepeated", testVirualOneRepeated),
        ("testVirualStrided", testVirualStrided),
        ("testVirualStridedMultiple", testVirualStridedMultiple),
        ("testVirualStridedMultiple2", testVirualStridedMultiple2),
        ("testVirualStridedMultiple3", testVirualStridedMultiple3),
        ("testVirualStridedMultiple4", testVirualStridedMultiple4),
        ("testVirualStridedMultiple5", testVirualStridedMultiple5),
        ("testVirualEmpty2", testVirualEmpty2),
        ("testScan", testScan),
        ("testShapeInit", testShapeInit),
    ]
}