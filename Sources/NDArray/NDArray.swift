

public struct NDArray<Scalar> {
    public let shape: [Int]

    public typealias ScalarGetter = ((Int, UnsafeMutableBufferPointer<Int>)) -> Scalar
    public typealias ScalarSetter = ((Int, UnsafeMutableBufferPointer<Int>), Scalar) -> Void
    public typealias ScalarGetterSetter = ((Int, UnsafeMutableBufferPointer<Int>), (Scalar) -> Scalar) -> Void

    public var anyNDArray: AnyNDArray<Scalar>

    public init<N:NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        shape = ndarray.shape
        anyNDArray = AnyNDArray(ndarray)
    }

    internal init(_ anyNDArray: AnyNDArray<Scalar>) {
        shape = anyNDArray.shape
        self.anyNDArray = anyNDArray
    }

    init(_ ndarray: NDArray) {
        self = ndarray
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

        let data = Ref(flatData)
        let arrayShape = ArrayShape(shape ?? calculatedShape)

        self = NDArray(BaseNDArray(data, shape: arrayShape))
    }

    public init(_ data: Scalar) {
        self = NDArray(ScalarNDArray(
            data,
            shape: []
        ))
    }

    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        anyNDArray.subscript_get(ranges)
    }

    public mutating func subscript_set(_ ranges: [ArrayRange], _ value: NDArray<Scalar>) -> NDArray<Scalar> {
        if !isKnownUniquelyReferenced(&anyNDArray) {
            self = anyNDArray.copy()
        }

        self = anyNDArray.subscript_set(ranges, value)

        return self
    }

    public func linearIndex(at indexes: [Int]) -> Int {
        anyNDArray.linearIndex(indexes)
    }

    public func dataValue(at indexes: [Int]) -> Scalar {
        anyNDArray.dataValue(indexes)
    }

    public func withScalarGetter(_ body: (@escaping ScalarGetter) -> Void) {
        anyNDArray.withScalarGetter(body)
    }

    public mutating func withScalarSetter(_ body: (@escaping ScalarSetter) -> Void) {
        anyNDArray.withScalarSetter(body)
    }

    public mutating func withScalarGetterSetter(_ body: (@escaping ScalarGetterSetter) -> Void) {
        anyNDArray.withScalarGetterSetter(body)
    }

    public func transposed(_ permutations: [Int]) -> NDArray<Scalar> {
        anyNDArray.transposed(permutations)
    }

    public func tiled(by amounts: [Int]) -> NDArray<Scalar> {
        anyNDArray.tiled(amounts)
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        anyNDArray.expandDimensions(axis)
    }

    public func scalarized() -> Scalar {
        anyNDArray.scalarized()
    }

    public func copy() -> NDArray<Scalar> {
        NDArray(anyNDArray.copy())
    }

    public func baseCopy() -> BaseNDArray<Scalar> {
        anyNDArray.baseCopy()
    }

    public func toArray<T: MultiArray>(_: T.Type) -> T {
        let cp: BaseNDArray = baseCopy()
        var array: [Any] = cp.data.value

        for n in cp.shape.reversed().dropLast() {
            array = array.chunked(into: n) as [Any]
        }

        return array as! T
    }
}

public class AnyNDArray<Scalar> {
    public let shape: [Int]

    let linearIndex: ([Int]) -> Int
    let dataValue: ([Int]) -> Scalar
    let subscript_get: ([ArrayRange]) -> NDArray<Scalar>
    let subscript_set: ([ArrayRange], NDArray<Scalar>) -> NDArray<Scalar>
    let withScalarGetter: ((@escaping NDArray<Scalar>.ScalarGetter) -> Void) -> Void
    let withScalarSetter: ((@escaping NDArray<Scalar>.ScalarSetter) -> Void) -> Void
    let withScalarGetterSetter: ((@escaping NDArray<Scalar>.ScalarGetterSetter) -> Void) -> Void
    let transposed: ([Int]) -> NDArray<Scalar>
    let tiled: ([Int]) -> NDArray<Scalar>
    let expandDimensions: (Int) -> NDArray<Scalar>
    let scalarized: () -> Scalar
    let copy: () -> NDArray<Scalar>
    let baseCopy: () -> BaseNDArray<Scalar>

    //
    @usableFromInline
    let isSetable: () -> Bool

    public init<N: NDArrayProtocol>(_ ndarray: N) where N.Scalar == Scalar {
        var ndarray = ndarray

        shape = ndarray.shape
        linearIndex = { ndarray.linearIndex(at: $0) }
        dataValue = { ndarray.dataValue(at: $0) }
        subscript_get = { ndarray.subscript_get($0) }
        subscript_set = {
            ndarray.subscript_set($0, $1)
        }
        withScalarGetter = { ndarray.withScalarGetter($0) }
        withScalarSetter = { ndarray.withScalarSetter($0) }
        withScalarGetterSetter = { ndarray.withScalarGetterSetter($0) }
        transposed = { ndarray.transposed($0) }
        tiled = { ndarray.tiled(by: $0) }
        expandDimensions = { ndarray.expandDimensions(axis: $0) }
        scalarized = { ndarray.scalarized() }
        copy = { ndarray.copy() }
        baseCopy = { ndarray.baseCopy() }
        isSetable = { ndarray is SetableNDArray }
    }
}