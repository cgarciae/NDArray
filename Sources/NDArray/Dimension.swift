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
    func realIndex(of: Int) -> Int
}

public protocol SqueezedDimension: DimensionProtocol {}

extension DimensionProtocol {
    @inlinable
    public func memoryStridedValue(of index: Int) -> Int {
        realIndex(of: index) * memory_layout.stride
    }
}

public struct Dimension: DimensionProtocol {
    public let length: Int
    public let memory_layout: MemoryLayout

    public init(length: Int, memory_stride: Int) {
        self.length = length
        memory_layout = MemoryLayout(length: length, stride: memory_stride)
    }

    @inlinable
    public func realIndex(of index: Int) -> Int { index }
}

public struct SingularDimension: DimensionProtocol {
    public let length: Int = 1
    public let memory_layout: MemoryLayout

    public init() {
        memory_layout = MemoryLayout(length: 1, stride: 0)
    }

    @inlinable
    public func realIndex(of index: Int) -> Int { 0 }
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
    public func realIndex(of index: Int) -> Int {
        return base.realIndex(of: start + index * stride)
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
    public func realIndex(of index: Int) -> Int {
        base.realIndex(of: length - 1 - index)
    }
}

public struct IndexedDimension: DimensionProtocol, SqueezedDimension {
    public let base: DimensionProtocol
    public let length: Int

    public let start: Int

    public var memory_layout: MemoryLayout

    fileprivate init(base: DimensionProtocol, start: Int) {
        self.base = base
        self.start = start

        memory_layout = base.memory_layout
        length = 1
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        base.realIndex(of: index + start)
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
    public func realIndex(of index: Int) -> Int {
        base.realIndex(of: index % base.length)
    }
}

extension DimensionProtocol {
    public func sliced(start: Int = 0, end: Int? = nil, stride: Int = 1) -> DimensionProtocol {
        var start = start
        var end = end ?? (stride > 0 ? length : 0)
        var length = self.length

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

    public func indexed(_ start: Int) -> DimensionProtocol {
        IndexedDimension(base: self, start: start)
    }

    public var inverted: DimensionProtocol {
        InvertedDimension(base: self)
    }
}