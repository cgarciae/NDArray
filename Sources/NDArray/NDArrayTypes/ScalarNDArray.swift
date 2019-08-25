public struct ScalarNDArray<Scalar>: NDArrayProtocol {
    public var data: Scalar
    public let shape: [Int]

    @inlinable
    public init(_ data: Scalar, shape: [Int]) {
        self.data = data
        self.shape = shape
    }

    public func subscript_get(_ ranges: [ArrayRange]) -> NDArray<Scalar> {
        var shape = self.shape
        var dimensionToBeRemoved = [Int]()
        var dimensionToBeAdded = [Int: Int]()

        for (i, range) in ranges.enumerated() {
            switch range {
            case .index:
                dimensionToBeRemoved.append(i)

            case let .slice(start: start, end: end, stride: stride):

                if start == 0, end == nil || end! == shape[i], stride == 1 {
                    continue
                }

                shape[i] = Dimension(length: shape[i], memory_stride: 0)
                    .sliced(
                        start: start,
                        end: end,
                        stride: stride
                    )
                    .length

            case let .filter(indexes):
                shape[i] -= indexes.count

            case .all:
                continue
            case .squeezeAxis:
                precondition(
                    shape[i] == 1,
                    "Cannot squeeze dimension \(i) of \(shape), expected 1 got \(shape[i])"
                )

                dimensionToBeRemoved.append(i)

            case .newAxis:
                dimensionToBeAdded[i] = 1

            case .ellipsis:
                fatalError("Ellipsis should be expand as a series of .all expressions")
            }
        }

        shape = shape
            .enumerated()
            .filter { i, d in !dimensionToBeRemoved.contains(i) }
            .map { i, d in d }

        for (i, dimension) in dimensionToBeAdded {
            shape.insert(dimension, at: i)
        }

        return NDArray(ScalarNDArray(
            data,
            shape: shape
        ))
    }

    public mutating func subscript_set(_ ranges: [ArrayRange], _ ndarray: NDArray<Scalar>) -> NDArray<Scalar> {
        if shape.isEmpty, ranges.isEmpty {
            data = ndarray.scalarized()
            return NDArray(self)
        }

        let ndarrayView = subscript_get(ranges)

        if ndarrayView.shape.isEmpty, ndarrayView.shape == [1] {
            data = ndarray.scalarized()
            return NDArray(self)
        }

        return NDArray(ndarrayView.baseCopy())
    }

    public func linearIndex(at indexes: [Int]) -> Int {
        0
    }

    public func dataValue(at indexes: [Int]) -> Scalar {
        data
    }

    public func withScalarGetter(_ body: (@escaping NDArray<Scalar>.ScalarGetter) -> Void) {
        body { _, rectIndex in
            self.data
        }
    }

    public mutating func withScalarGetterSetter(_ body: (@escaping NDArray<Scalar>.ScalarGetterSetter) -> Void) {
        var data = self.data
        defer { self.data = data }

        body { indexer, f in
            data = f(data)
        }
    }

    public mutating func withScalarSetter(_ body: (@escaping NDArray<Scalar>.ScalarSetter) -> Void) {
        var data = self.data
        defer { self.data = data }

        body { indexer, value in
            data = value
        }
    }

    // ops
    public func transposed(_ indexes: [Int]) -> NDArray<Scalar> {
        NDArray(ScalarNDArray(
            data,
            shape: indexes.map { i in shape[i] }
        ))
    }

    public func tiled(by repetitions: [Int]) -> NDArray<Scalar> {
        NDArray(ScalarNDArray(
            data,
            shape: zip(shape, repetitions).map(*)
        ))
    }

    public func expandDimensions(axis: Int) -> NDArray<Scalar> {
        var shape = self.shape
        shape.insert(1, at: axis)

        return NDArray(ScalarNDArray(
            data,
            shape: shape
        ))
    }

    public func scalarized() -> Scalar {
        data
    }

    public func copy() -> NDArray<Scalar> {
        if shape.isEmpty {
            return NDArray(ScalarNDArray(data, shape: shape))
        } else {
            return NDArray(baseCopy())
        }
    }
}