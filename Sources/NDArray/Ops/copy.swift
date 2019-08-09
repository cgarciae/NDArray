
extension NDArray {
    public func copy() -> NDArray {
        let nElements = shape.product()

        let arrayC = [Scalar](unsafeUninitializedCapacity: nElements) { arrayC, count in
            count = nElements

            self.data.value.withUnsafeBufferPointer { arrayA in
                for index in indexSequence(range: 0 ..< nElements, shape: shape) {
                    arrayC[index.linearIndex] = arrayA[self.linearIndex(at: index.rectangularIndex)]
                }
            }
        }

        return NDArray(
            arrayC,
            shape: shape
        )
    }
}