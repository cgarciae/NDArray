
public struct NDArray<Scalar> {
    public let data: [Scalar]
    @inlinable public var shape: [Int] { array_shape.virtualShape }
    @usableFromInline internal let array_shape: Shape

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

        self.data = flatData
        array_shape = Shape(shape ?? calculatedShape)
    }

    @usableFromInline
    internal init(_ data: [Scalar], shape: Shape) {
        self.data = data
        array_shape = shape
    }

    public init(scalar data: Scalar) {
        self.data = [data]
        array_shape = Shape([DimensionProtocol]())
    }

    @inlinable
    public func realIndex(of index: Int) -> Int {
        array_shape.realIndex(of: index)
    }

    @inlinable
    public func dataValue(at index: Int) -> Scalar {
        let realIndex = array_shape.realIndex(of: index)
        return data[realIndex]
    }
}