import Foundation

public struct MemoryLayout {
    public let stride: Int
    public let length: Int

    fileprivate init(length: Int, stride: Int) {
        self.stride = stride
        self.length = length
    }
}

public protocol DimensionProtocol {
    var length: Int { get }
    var memory_layout: MemoryLayout { get }

    @inlinable
    func linearIndex(of: Int) -> Int
}

// public protocol SqueezedDimension: DimensionProtocol {}
public protocol UnmodifiedDimension: DimensionProtocol {}

extension DimensionProtocol {
    @inlinable
    public func strideValue(of index: Int) -> Int {
        linearIndex(of: index) * memory_layout.stride
    }
}

public struct Dimension: DimensionProtocol, UnmodifiedDimension {
    public let length: Int
    public let memory_layout: MemoryLayout

    public init(length: Int, memory_stride: Int) {
        self.length = length
        memory_layout = MemoryLayout(length: length, stride: memory_stride)
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int { index }
}

public struct SingularDimension: DimensionProtocol, UnmodifiedDimension {
    public let length: Int = 1
    public let memory_layout: MemoryLayout

    public init() {
        memory_layout = MemoryLayout(length: 1, stride: 0)
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

    public var memory_layout: MemoryLayout

    fileprivate init(base: DimensionProtocol, start: Int, end: Int, stride: Int) {
        self.base = base
        self.stride = stride
        self.start = start
        self.end = end

        if abs(end - start) % abs(stride) != 0 {
            length = (1 + abs(end - start) / abs(stride))
        } else {
            length = (abs(end - start) / abs(stride))
        }

        memory_layout = base.memory_layout
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        return base.linearIndex(of: start + index * stride)
    }
}

public struct InvertedDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public let length: Int

    public var memory_layout: MemoryLayout

    fileprivate init(base: DimensionProtocol) {
        self.base = base

        length = base.length
        memory_layout = base.memory_layout
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        base.linearIndex(of: length - 1 - index)
    }
}

public struct TiledDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public var length: Int
    public var repetitions: Int
    public var memory_layout: MemoryLayout

    fileprivate init(base: DimensionProtocol, repetitions: Int) {
        self.base = base
        self.repetitions = repetitions

        length = base.length * repetitions
        memory_layout = base.memory_layout
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        base.linearIndex(of: index % base.length)
    }
}

public struct FilteredDimension: DimensionProtocol {
    public let base: DimensionProtocol
    public var length: Int
    public var memory_layout: MemoryLayout
    public var indexes: [Int]

    fileprivate init(base: DimensionProtocol, indexes: [Int]) {
        self.base = base
        self.indexes = indexes.map { index in
            index < 0 ? index + base.length : index
        }

        length = indexes.count
        memory_layout = base.memory_layout
    }

    @inlinable
    public func linearIndex(of index: Int) -> Int {
        base.linearIndex(of: indexes[index])
    }
}

extension DimensionProtocol {
    public func sliced(start: Int? = nil, end: Int? = nil, stride: Int = 1) -> DimensionProtocol {
        var start = start ?? (stride > 0 ? 0 : -1)
        var end = end ?? (stride > 0 ? length : 0)

        if start < 0 {
            start += length
        }
        if end < 0 {
            end += length
        }

        if stride < 0 {
            end -= 1
        }

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