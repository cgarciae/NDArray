
extension NDArray: CustomStringConvertible {
    public var description: String {
        let nElements = shape.product()
        var s = "\(Self.self)\(shape)(" + String(repeating: "[", count: max(shape.count - 1, 0))

        withScalarGetter { valueAt in

            if shape.count == 0 {
                s += "\(dataValue(at: [Int]()))" + ")"
            } else if shape.count == 1 {
                var arrayString = ""
                for index in indexSequence(range: 0 ..< nElements, shape: shape) {
                    let (i, _) = index
                    arrayString += "\(valueAt(index))" + (i + 1 != nElements ? ", " : "")
                }
                s += "[\(arrayString)])"
            } else {
                let reversedShape = Array(shape.reversed())
                let lastDim = reversedShape[0]
                let secondLastDim = reversedShape[1]

                s += "\n"

                var arrayString = ""

                for index in indexSequence(range: 0 ..< nElements + 1, shape: shape) {
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
                s += String(repeating: "]", count: max(shape.count - 1, 1)) + ")"
            }
        }

        return s
    }
}