// import Foundation
@testable import NDArray
import XCTest

final class DimensionTests: XCTestCase {
    func testVirualEmpty() {
        let dimension = SingularDimension()

        let linearIndex = dimension.linearIndex(of: 0)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 1)
    }

    func testVirualOneRepeated() {
        let dimension = SingularDimension().tiled(3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 3)
    }

    func testVirualStrided() {
        let dimension = SingularDimension().tiled(10).sliced(start: 0, end: 10, stride: 3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 4)
    }

    func testVirualStridedMultiple() {
        let dimension = SingularDimension().tiled(4).sliced(start: 0, end: 4, stride: 3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 2)
    }

    func testVirualStridedMultiple2() {
        let dimension = SingularDimension().tiled(7).sliced(start: 0, end: 7, stride: 3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 3)
    }

    func testVirualStridedMultiple3() {
        let dimension = SingularDimension().tiled(8).sliced(start: 0, end: 8, stride: 3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 3)
    }

    func testVirualStridedMultiple4() {
        let dimension = SingularDimension().tiled(9).sliced(start: 0, end: 9, stride: 3)

        let linearIndex = dimension.linearIndex(of: 1)
        let count = dimension.length

        XCTAssertEqual(linearIndex, 0)
        XCTAssertEqual(count, 3)
    }

    func testVirualStridedMultiple5() {
        let dimension = SingularDimension().tiled(9).sliced(start: 5, end: 6, stride: 3)

        XCTAssertEqual(dimension.linearIndex(of: 0), 0)
        XCTAssertEqual(dimension.length, 1)
    }

    func testVirualEmpty2() {
        let dimension = SingularDimension().tiled(3)

        let linearIndex = dimension.linearIndex(of: 1)

        XCTAssertEqual(linearIndex, 0)
    }

    func testScan() {
        let accumulatedProduct = [1, 2, 3, 4].scan(*)

        let total = accumulatedProduct.sum()

        XCTAssertEqual(total, 33)
    }

    // func testShapeInit() {
    //     let shape = ArrayShape([5, 1, 4, 2])

    //     XCTAssertEqual(
    //         shape.dimensions.map { $0.length },
    //         [8, 0, 2, 1]
    //     )
    // }

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
        // ("testShapeInit", testShapeInit),
    ]
}