import Foundation

// public struct MemoryLayout {
//     public let stride: Int
//     public let length: Int

//     fileprivate init(length: Int, stride: Int) {
//         self.stride = stride
//         self.length = length
//     }
// }

public protocol DimensionProtocol {
    var length: Int { get }
    // var memory_stride: Int { get }

    @inlinable
    func linearIndex(of: Int) -> Int
}

// public protocol SqueezedDimension: DimensionProtocol {}
public protocol UnmodifiedDimension: DimensionProtocol {}

public struct Dimension: DimensionProtocol, UnmodifiedDimension {
    public let length: Int
    // public let memory_stride: Int

    public init(length: Int) {
        self.length = length
        // self.memory_stride = memory_stride
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int { index }
}

public struct SingularDimension: DimensionProtocol, UnmodifiedDimension {
    public let length: Int = 1
    // public let memory_stride: Int

    public init() {
        // memory_stride = 0
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int { 0 }
}

public struct SlicedDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public let length: Int

    public let stride: Int
    public let start: Int
    public let end: Int

    // public var memory_stride: Int

    fileprivate init(base: DimensionProtocol, start: Int, end: Int, stride: Int) {
        self.base = base
        self.stride = stride
        self.start = start
        self.end = end

        // length = (1 + abs(end - start) / abs(stride))
        switch abs(end - start).quotientAndRemainder(dividingBy: abs(stride)) {
        case let (quotient, 0):
            length = quotient
        case let (quotient, _):
            length = quotient + 1
        }

        // if abs(end - start) % abs(stride) != 0 {
        //     length = (1 + abs(end - start) / abs(stride))
        // } else {
        //     length = (abs(end - start) / abs(stride))
        // }
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        return base.linearIndex(of: start + index * stride)
    }
}

// public struct InvertedDimension: DimensionProtocol {
//     public let base: DimensionProtocol
//     public let length: Int

//     public var memory_stride: Int

//     fileprivate init(base: DimensionProtocol) {
//         self.base = base

//         length = base.length
//         memory_stride = base.memory_stride
//     }

//     @inlinable
//     public func linearIndex(of index: Int) -> Int {
//         base.linearIndex(of: length - 1 - index)
//     }
// }

public struct TiledDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public var length: Int
    public var repetitions: Int
    // public var memory_stride: Int

    fileprivate init(base: DimensionProtocol, repetitions: Int) {
        self.base = base
        self.repetitions = repetitions

        length = base.length * repetitions
        // memory_stride = base.memory_stride
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        base.linearIndex(of: index % base.length)
    }
}

public struct FilteredDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public var length: Int
    // public var memory_stride: Int
    public var indexes: [Int]

    fileprivate init(base: DimensionProtocol, indexes: [Int]) {
        self.base = base
        self.indexes = indexes

        length = indexes.count
        // memory_stride = base.memory_stride
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        base.linearIndex(of: indexes[index])
    }
}

extension DimensionProtocol {
    public func sliced(start: Int, end: Int, stride: Int) -> DimensionProtocol {
        return SlicedDimension(
            base: self,
            start: start,
            end: end,
            stride: stride
        )
    }

    public func tiled(_ repetitions: Int) -> DimensionProtocol {
        TiledDimension(base: self, repetitions: repetitions)
    }

    public func select(indexes: [Int]) -> DimensionProtocol {
        FilteredDimension(base: self, indexes: indexes)
    }
}