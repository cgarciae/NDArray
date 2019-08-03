
@usableFromInline
internal final class Ref<A> {
    @usableFromInline var value: A

    @usableFromInline init(_ value: A) {
        self.value = value
    }
}

public struct NDArray<Scalar> {
    @usableFromInline internal var data: Ref<[Scalar]>

    @inlinable public var shape: [Int] { arrayShape.virtualShape }
    @usableFromInline internal var arrayShape: ArrayShape

    @usableFromInline
    internal init(_ data: Ref<[Scalar]>, shape: ArrayShape) {
        self.data = data
        arrayShape = shape
    }

    public init(_ data: [Any], shape: [Int]? = nil) {
        let (flatData, calculatedShape): ([Scalar], [Int]) = flattenArrays(data)

        precondition(
            calculatedShape.reduce(1, *) == flatData.count,
            "All sub-arrays in data must have equal length. Calculated shape: \(calculatedShape), \(flatData)"
        )

        if let shape = shape {
            precondition(
                shape.reduce(1, *) == flatData.count,
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
        arrayShape = ArrayShape([DimensionProtocol]())
        self.data = Ref([data])
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        arrayShape.realIndex(of: index)
    }

    @inlinable
    public func dataValue(at index: Int) -> Scalar {
        let realIndex = arrayShape.realIndex(of: index)
        return data.value[realIndex]
    }
}