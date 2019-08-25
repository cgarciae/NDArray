
extension NDArray: CustomStringConvertible {
    public var description: String {
        let nElements = shape.product()
        var s = "\(Self.self)\(shape)(" + String(repeating: "[", count: Swift.max(shape.count - 1, 0))

        withScalarGetter { valueAt in

            if self.shape.count == 0 {
                s += "\(self.dataValue(at: [Int]()))" + ")"
            } else if self.shape.count == 1 {
                var arrayString = ""
                for index in indexSequence(range: 0 ..< nElements, shape: self.shape) {
                    let (i, _) = index
                    arrayString += "\(valueAt(index))" + (i + 1 != nElements ? ", " : "")
                }
                s += "[\(arrayString)])"
            } else {
                let reversedShape = Array(self.shape.reversed())
                let lastDim = reversedShape[0]
                let secondLastDim = reversedShape[1]

                s += "\n"

                var arrayString = ""

                for index in indexSequence(range: 0 ..< nElements + 1, shape: self.shape) {
                    let (i, _) = index

                    if i % lastDim == 0, i > 0 {
                        s += "    [\(arrayString)],\n"
                        arrayString = ""
                    }

                    if i % (lastDim * secondLastDim) == 0, i > 0, i < nElements {
                        s += "\n"
                    }

                    if i < nElements {
                        arrayString += "\(valueAt(index))" + ((i + 1) % lastDim != 0 ? ", " : "")
                    }
                }
                s += String(repeating: "]", count: Swift.max(self.shape.count - 1, 1)) + ")"
            }
        }

        return s
    }
}