
extension NDArray: CustomStringConvertible {
    public var description: String {
        let nElements = shape.product()
        var s = "\(Self.self)\(shape)(" + String(repeating: "[", count: max(shape.count - 1, 0))

        if shape.count == 0 {
            return s + "\(dataValue(at: []))" + ")"
        } else if shape.count == 1 {
            var arrayString = ""
            for (i, rectangularIndex) in indexSequence(range: 0 ..< nElements, shape: shape) {
                arrayString += "\(dataValue(at: rectangularIndex.value))" + (i + 1 != nElements ? ", " : "")
            }
            return s + "[\(arrayString)])"
        } else {
            let reversedShape = Array(shape.reversed())
            let lastDim = reversedShape[0]
            let secondLastDim = reversedShape[1]

            s += "\n"

            var arrayString = ""

            for (i, rectangularIndex) in indexSequence(range: 0 ..< nElements + 1, shape: shape) {
                if i % lastDim == 0, i > 0 {
                    s += "    [\(arrayString)],\n"
                    arrayString = ""
                }

                if i % (lastDim * secondLastDim) == 0, i > 0, i < nElements {
                    s += "\n"
                }

                if i < nElements {
                    arrayString += "\(dataValue(at: rectangularIndex.value))" + ((i + 1) % lastDim != 0 ? ", " : "")
                }
            }
            return s + String(repeating: "]", count: max(shape.count - 1, 1)) + ")"
        }
    }
}