
extension NDArrayProtocol {
    internal func copy() -> BaseNDArray<Scalar> {
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
            arrayC,
            shape: shape
        )
    }

    public func copy() -> NDArray<Scalar> {
        let cp: BaseNDArray = copy()
        return NDArray(cp)
    }
}