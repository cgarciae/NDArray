// import Foundation
@testable import NDArray
import XCTest

final class DimensionTests: XCTestCase {
    func testVirualEmpty() {
        let dimension = Dimension(start: 0, end: 0, total: 0, stride: 1, repetitions: 3, memory_stride: 0)

        // let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        // XCTAssertEqual(realIndex, 0)
        XCTAssertEqual(count, 0)
        XCTAssertEqual(realCount, 0)
    }

    func testVirualOneRepeated() {
        let dimension = Dimension(start: 0, end: 1, total: 1, stride: 1, repetitions: 3, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 0)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 1)
    }

    func testVirualStrided() {
        let dimension = Dimension(start: 0, end: 10, total: 10, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 4)
        XCTAssertEqual(realCount, 10)
    }

    func testVirualStridedMultiple() {
        let dimension = Dimension(start: 0, end: 4, total: 4, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 2)
        XCTAssertEqual(realCount, 4)
    }

    func testVirualStridedMultiple2() {
        let dimension = Dimension(start: 0, end: 7, total: 7, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 7)
    }

    func testVirualStridedMultiple3() {
        let dimension = Dimension(start: 0, end: 8, total: 8, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 8)
    }

    func testVirualStridedMultiple4() {
        let dimension = Dimension(start: 0, end: 9, total: 9, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 3)
        XCTAssertEqual(count, 3)
        XCTAssertEqual(realCount, 9)
    }

    func testVirualStridedMultiple5() {
        let dimension = Dimension(start: 5, end: 6, total: 9, stride: 3, repetitions: 1, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 0)
        let count = dimension.count
        let realCount = dimension.realCount

        XCTAssertEqual(realIndex, 5)
        XCTAssertEqual(count, 1)
        XCTAssertEqual(realCount, 9)
    }

    func testVirualEmpty2() {
        let dimension = Dimension(start: 0, end: 1, total: 0, stride: 1, repetitions: 3, memory_stride: 0)

        let realIndex = dimension.realIndex(of: 1)

        XCTAssertEqual(realIndex, 0)
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
    ]
}