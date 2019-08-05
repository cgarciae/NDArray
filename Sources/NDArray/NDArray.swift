

public final class Ref<A>: CustomStringConvertible {
    @usableFromInline var value: A

    @usableFromInline init(_ value: A) {
        self.value = value
    }

    public var description: String { "\(value)" }
}

public struct NDArray<Scalar> {
    @usableFromInline internal var data: Ref<[Scalar]>
    @usableFromInline internal var arrayShape: ArrayShape

    public var shape: [Int] { arrayShape.dimensionLengths }

    @usableFromInline
    internal init(_ data: Ref<[Scalar]>, shape: ArrayShape) {
        self.data = data
        arrayShape = shape
    }

    public init(_ data: [Any], shape: [Int]? = nil) {
        let (flatData, calculatedShape): ([Scalar], [Int]) = flattenArrays(data)

        precondition(
            calculatedShape.product() == flatData.count,
            "All sub-arrays in data must have equal length. Calculated shape: \(calculatedShape), \(flatData)"
        )

        if let shape = shape {
            precondition(
                shape.product() == flatData.count,
                "Invalid shape, number of elements"
            )
        }

        arrayShape = ArrayShape(shape ?? calculatedShape)
        self.data = Ref(flatData)
    }

    @usableFromInline
    internal init(_ data: [Scalar], shape: ArrayShape) {
        arrayShape = shape
        self.data = Ref(data)
    }

    public init(_ data: Scalar) {
        arrayShape = ArrayShape([DimensionProtocol](), linearMemoryOffset: 0)
        self.data = Ref([data])
    }

    @inlinable
    public func linearIndex(at indexes: [Int]) -> Int {
        arrayShape.linearIndex(of: indexes)
    }

    @inlinable
    public func dataValue(at indexes: [Int]) -> Scalar {
        let linearIndex = arrayShape.linearIndex(of: indexes)
        return data.value[linearIndex]
    }
}