
extension NDArray {
    public func copy() -> NDArray {
        let nElements = shape.reduce(1, *)

        let arrayC = [Scalar](unsafeUninitializedCapacity: nElements) { arrayC, count in
            count = nElements

            self.data.value.withUnsafeBufferPointer { arrayA in
                for i in 0 ..< nElements {
                    arrayC[i] = arrayA[self.realIndex(of: i)]
                }
            }
        }

        return NDArray(
            arrayC,
            shape: shape
        )
    }
}