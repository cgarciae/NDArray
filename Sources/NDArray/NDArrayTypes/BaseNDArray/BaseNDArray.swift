

public struct BaseNDArray<Scalar>: NDArrayProtocol, SetableNDArray {
    public var data: Ref<[Scalar]>
    public var arrayShape: ArrayShape
    public var memory_strides: [Int]
    public var memory_offset: Int
    public var shape: [Int] { arrayShape.shape }

    @inlinable
    public init(_ data: Ref<[Scalar]>, shape: ArrayShape, memory_offset: Int, memory_strides: [Int]?) {
        self.data = data
        self.memory_offset = memory_offset
        arrayShape = shape
        self.memory_strides = memory_strides ?? getDimensionStrides(of: shape.shape)
    }

    // public init(_ data: [Any], shape: [Int]? = nil) {
    //     let (flatData, calculatedShape): ([Scalar], [Int]) = flattenArrays(data)

    //     precondition(
    //         calculatedShape.product() == flatData.count,
    //         "All sub-arrays in data must have equal length. Calculated shape: \(calculatedShape), \(flatData)"
    //     )

    //     if let shape = shape {
    //         precondition(
    //             shape.product() == flatData.count,
    //             "Invalid shape, number of elements"
    //         )
    //     }

    //     arrayShape = ArrayShape(shape ?? calculatedShape)
    //     self.data = Ref(flatData)
    // }

    // @usableFromInline
    // internal init(_ data: [Scalar], shape: ArrayShape) {
    //     arrayShape = shape
    //     self.data = Ref(data)
    // }

    // public init(_ data: Scalar) {
    //     arrayShape = ArrayShape([DimensionProtocol](), memory_offset: 0)
    //     self.data = Ref([data])
    // }

    @inlinable
    public func linearIndex(at indexes: UnsafeMutableBufferPointer<Int>) -> Int {
        var partialIndex = 0

        for i in 0 ..< indexes.count {
            let index = indexes[i]
            let dimension = arrayShape.dimensions[i]
            let memory_stride = memory_strides[i]

            partialIndex += dimension.linearIndex(of: index) * memory_stride
        }

        return partialIndex + memory_offset
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
            linearIndex(at: indexes)
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
                let index = self.linearIndex(at: rectIndex)
                return data[index]
            }
        }
    }

    @inlinable
    public mutating func withScalarSetter(_ body: (@escaping NDArray<Scalar>.ScalarSetter) -> Void) {
        data.value.withUnsafeMutableBufferPointer { dataIn in
            var data = dataIn
            defer { dataIn = data }

            body { [self] indexer, value in
                let (_, rectangularIndex) = indexer
                let index = self.linearIndex(at: rectangularIndex)

                data[index] = value
            }
        }
    }

    @inlinable
    public mutating func withScalarGetterSetter(_ body: (@escaping NDArray<Scalar>.ScalarGetterSetter) -> Void) {
        data.value.withUnsafeMutableBufferPointer { dataIn in
            var data = dataIn
            defer { dataIn = data }

            body { [self] index, f in
                let index = self.linearIndex(at: index.1)
                let value = f(data[index])

                data[index] = value
            }
        }
    }

    @inlinable
    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        var memory_offset = self.memory_offset
        var memory_strides = self.memory_strides

        for (i, range) in ranges.enumerated().reversed() {
            switch range {
            case let .index(index):
                memory_offset += self.arrayShape.dimensions[i].linearIndex(of: index) * memory_strides[i]
                memory_strides.remove(at: i)

            case .squeezeAxis:
                memory_offset += self.arrayShape.dimensions[i].linearIndex(of: 0) * memory_strides[i]
                memory_strides.remove(at: i)

            default:
                continue
            }
        }

        let arrayShape = self.arrayShape[ranges]

        let ndarray = BaseNDArray(
            data,
            shape: arrayShape,
            memory_offset: memory_offset,
            memory_strides: memory_strides
        )

        if ndarray.shape.isEmpty {
            return NDArray(ScalarNDArray(
                ndarray.scalarized(),
                shape: []
            ))
        } else {
            return NDArray(ndarray)
        }
    }

    public mutating func subscript_set(_ ranges: [ArrayRange], _ ndarray: NDArray<Scalar>) -> NDArray<Scalar> {
        var ndarray = ndarray

        if !isKnownUniquelyReferenced(&data) {
            self = baseCopy()
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

        return NDArray(self)
    }

    public func tiled(by repetitions: [Int]) -> NDArray<Scalar> {
        var dimensions = arrayShape.dimensions

        for i in 0 ..< dimensions.count {
            if repetitions[i] > 1 {
                dimensions[i] = dimensions[i].tiled(repetitions[i])
            }
        }

        return NDArray(BaseNDArray(
            data,
            shape: ArrayShape(
                dimensions
            ),
            memory_offset: memory_offset,
            memory_strides: memory_strides
        ))
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        var dimensions = arrayShape.dimensions

        dimensions.insert(SingularDimension(), at: axis)

        return NDArray(BaseNDArray(
            data,
            shape: ArrayShape(
                dimensions
            ),
            memory_offset: memory_offset,
            memory_strides: memory_strides
        ))
    }

    public func scalarized() -> Scalar {
        precondition(shape == [] || shape == [1], "Cannot convert non-scalar NDArray to scalar, got shape \(shape)")

        return dataValue(at: [])
    }

    public func copy() -> NDArray<Scalar> {
        let nElements = shape.product()

        let arrayC = [Scalar](unsafeUninitializedCapacity: nElements) { arrayC, count in
            count = nElements

            self.withScalarGetter { arrayAScalarGetter in
                for index in indexSequence(range: 0 ..< nElements, shape: shape) {
                    arrayC[index.linearIndex] = arrayAScalarGetter(index)
                }
            }
        }

        return NDArray(BaseNDArray(
            Ref(arrayC),
            shape: ArrayShape(shape),
            memory_offset: 0,
            memory_strides: nil
        ))
    }

    public func transposed(_ indexes: [Int]) -> NDArray<Scalar> {
        precondition(shape.count >= indexes.count)

        return NDArray(BaseNDArray(
            data,
            shape: ArrayShape(
                indexes.map { i in arrayShape.dimensions[i] }
            ),
            memory_offset: memory_offset,
            memory_strides: indexes.map { i in memory_strides[i] }
        ))
    }
}

extension NDArrayProtocol {
    internal func baseCopy() -> BaseNDArray<Scalar> {
        let nElements = shape.product()

        let arrayC = [Scalar](unsafeUninitializedCapacity: nElements) { arrayC, count in
            count = nElements

            self.withScalarGetter { arrayAScalarGetter in
                for index in indexSequence(range: 0 ..< nElements, shape: shape) {
                    arrayC[index.linearIndex] = arrayAScalarGetter(index)
                }
            }
        }

        return BaseNDArray(
            Ref(arrayC),
            shape: ArrayShape(shape),
            memory_offset: 0,
            memory_strides: nil
        )
    }
}