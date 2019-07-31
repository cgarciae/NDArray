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

    @inlinable
    public var memory_layout: MemoryLayout { base.memory_layout }

    fileprivate init(base: DimensionProtocol, start: Int, end: Int, stride: Int) {
        self.base = base
        self.start = start
        self.end = min(end, base.length)
        self.stride = stride

        if (end - start) % stride != 0 {
            length = (1 + (end - start) / stride)
        } else {
            length = ((end - start) / stride)
        }
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        base.realIndex(of: index * stride + start)
    }
}

public struct IndexedDimension: DimensionProtocol, SqueezedDimension {
    public let base: DimensionProtocol
    public let length: Int

    public let start: Int

    @inlinable
    public var memory_layout: MemoryLayout { base.memory_layout }

    fileprivate init(base: DimensionProtocol, start: Int) {
        self.base = base
        self.start = start

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
    @inlinable
    public var memory_layout: MemoryLayout { base.memory_layout }

    fileprivate init(base: DimensionProtocol, repetitions: Int) {
        self.base = base
        self.repetitions = repetitions

        length = base.length * repetitions
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        base.realIndex(of: index % base.length)
    }
}

extension DimensionProtocol {
    public func sliced(start: Int = 0, end: Int? = nil, stride: Int = 1) -> DimensionProtocol {
        SlicedDimension(
            base: self,
            start: start,
            end: end ?? length,
            stride: stride
        )
    }

    public func tiled(_ repetitions: Int) -> DimensionProtocol {
        TiledDimension(base: self, repetitions: repetitions)
    }

    public func indexed(_ start: Int) -> DimensionProtocol {
        IndexedDimension(base: self, start: start)
    }
}