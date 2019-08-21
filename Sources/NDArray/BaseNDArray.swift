
public struct BaseNDArray<Scalar> :NDArrayProtocol {
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
    public func linearIndex(at indexes: UnsafeMutableBufferPointer<Int>) -> Int {
        arrayShape.linearIndex(of: indexes)
    }

    @inlinable
    public func linearIndex(at indexes: [Int]) -> Int {
        var indexes = indexes

        return indexes.withUnsafeMutableBufferPointer { indexes in
            linearIndex(at: indexes)
        }
    }

    @inlinable
    public func dataValue(at indexes: UnsafeMutableBufferPointer<Int>) -> Scalar {
        return data.value[
            arrayShape.linearIndex(of: indexes)
        ]
    }

    @inlinable
    public func dataValue(at indexes: [Int]) -> Scalar {
        var indexes = indexes

        return indexes.withUnsafeMutableBufferPointer { indexes in
            dataValue(at: indexes)
        }
    }

    @inlinable
    public func withScalarGetter(_ body: (@escaping NDArray<Scalar>.ScalarGetter) -> Void) {
        data.value.withUnsafeBufferPointer { data in

            body { _, rectIndex in
                let index = self.arrayShape.linearIndex(of: rectIndex)
                return data[index]
            }
        }
    }

    @inlinable
    public mutating func withScalarSetter(_ body: (@escaping NDArray<Scalar>.ScalarSetter) -> Void) {
        if isKnownUniquelyReferenced(&data) {
            print("UNIQUE")
        } else {
            print("NOT UNIQUE")
        }
        data.value.withUnsafeMutableBufferPointer { dataIn in
            var data = dataIn
            defer { dataIn = data }

            body { [self] indexer, value in
                let (_, rectangularIndex) = indexer
                let index = self.arrayShape.linearIndex(of: rectangularIndex)

                data[index] = value
            }
        }
    }

    @inlinable
    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        var ranges = ranges

        let nEllipsis = ranges.filter(isEllipsis).count

        precondition(nEllipsis <= 1, "A maximum of 1 .ellipsis can be used, got \(ranges)")

        if nEllipsis == 1 {
            let ellipsisIndex = ranges.firstIndex(where: isEllipsis)!
            let nAll = 1 + shape.count - ranges.count

            ranges.remove(at: ellipsisIndex)

            for _ in 0 ..< nAll {
                ranges.insert(.all, at: ellipsisIndex)
            }
        }

        precondition(shape.count >= ranges.count)

        var dimensions = arrayShape.dimensions
        var linearMemoryOffset = arrayShape.linearMemoryOffset
        var dimensionToBeRemoved = [Int]()
        var dimensionToBeAdded = [Int: DimensionProtocol]()

        for (i, range) in ranges.enumerated() {
            switch range {
            case let .index(index):
                let index = index < 0 ? dimensions[i].length + index : index

                linearMemoryOffset += dimensions[i].strideValue(of: index)
                dimensionToBeRemoved.append(i)

            case let .slice(start: start, end: end, stride: stride):

                if start == 0, end == nil || end! == dimensions[i].length, stride == 1 {
                    continue
                }

                dimensions[i] = dimensions[i].sliced(
                    start: start,
                    end: end,
                    stride: stride
                )
            case let .filter(indexes):
                dimensions[i] = dimensions[i].select(indexes: indexes)

            case .all:
                continue
            case .squeezeAxis:
                precondition(
                    dimensions[i].length == 1,
                    "Cannot squeeze dimension \(i) of \(shape), expected 1 got \(shape[i])"
                )

                linearMemoryOffset += dimensions[i].strideValue(of: 0)
                dimensionToBeRemoved.append(i)

            case .newAxis:
                dimensionToBeAdded[i] = SingularDimension()

            case .ellipsis:
                fatalError("Ellipsis should be expand as a series of .all expressions")
            }
        }

        // TODO: this implementation is not correct due the fact the the length of dimension is changing
        // A correct way to implement this would be to do the operations sorted by the index
        // from high to low.
        dimensions = dimensions
            .enumerated()
            .filter { i, d in !dimensionToBeRemoved.contains(i) }
            .map { i, d in d }

        for (i, dimension) in dimensionToBeAdded {
            dimensions.insert(dimension, at: i)
        }

        return NDArray(
            BaseNDArray(
                data,
                shape: ArrayShape(
                    dimensions,
                    linearMemoryOffset: linearMemoryOffset
                )
            )
        )
    }

    public mutating func subscript_set(_ ranges: [ArrayRange], _ ndarray: NDArray<Scalar>) {
        var ndarray = ndarray

        let allAreUnmodifiedlDimensions = arrayShape
            .dimensions.lazy
            .map { $0 is UnmodifiedDimension }
            .reduce(true) { $0 && $1 }

        if !isKnownUniquelyReferenced(&data) {
            let cp: BaseNDArray = copy()
            data = cp.data
            arrayShape = cp.arrayShape
        }

        if !isKnownUniquelyReferenced(&data) {
            print("AAA UNIQUE")
        } else {
            print("AAA NOT UNIQUE")
        }

        var ndarrayView = subscript_get(ranges)
        let nElements = ndarrayView.shape.product()

        if ndarrayView.shape != ndarray.shape {
            (ndarrayView, ndarray) = broadcast(ndarrayView, and: ndarray)
        }

        let ndarrayViewShape = ndarrayView.shape

        ndarrayView.withScalarSetter { viewScalarSetter in
            ndarray.withScalarGetter { ndarrayScalarGetter in
                for index in indexSequence(range: 0 ..< nElements, shape: ndarrayViewShape) {
                    let value = ndarrayScalarGetter(index)
                    viewScalarSetter(index, value)
                }
            }
        }
    }

    public func tiled(by repetitions: [Int]) -> NDArray<Scalar> {
        var dimensions = arrayShape.dimensions

        for i in 0 ..< dimensions.count {
            if repetitions[i] > 1 {
                dimensions[i] = dimensions[i].tiled(repetitions[i])
            }
        }

        return NDArray(BaseNDArray(
            data.value,
            shape: ArrayShape(
                dimensions,
                linearMemoryOffset: arrayShape.linearMemoryOffset
            )
        ))
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        var dimensions = arrayShape.dimensions

        dimensions.insert(SingularDimension(), at: axis)

        return NDArray(BaseNDArray(
            data.value,
            shape: ArrayShape(
                dimensions,
                linearMemoryOffset: arrayShape.linearMemoryOffset
            )
        ))
    }

    public func scalarized() -> Scalar {
        precondition(shape == [], "Cannot convert non-scalar NDArray to scalar, got shape \(shape)")

        return data.value[0]
    }

    public mutating func copyInternals() {
        if !isKnownUniquelyReferenced(&data) {
            let cp: BaseNDArray = copy()
            data = cp.data
            arrayShape = cp.arrayShape
        }
    }
}