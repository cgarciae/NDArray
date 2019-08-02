
extension NDArray: CustomStringConvertible {
    public var description: String {
        let nElements = shape.reduce(1, *)
        var s = "\(Self.self)\(shape)(" + String(repeating: "[", count: max(shape.count - 1, 0))

        if shape.count == 0 {
            return s + "\(dataValue(at: 0))" + ")"
        } else if shape.count == 1 {
            var arrayString = ""
            for i in 0 ..< nElements {
                arrayString += "\(dataValue(at: i))" + (i + 1 != nElements ? ", " : "")
            }
            return s + "[\(arrayString)])"
        } else {
            let reversedShape = Array(shape.reversed())
            let lastDim = reversedShape[0]
            let secondLastDim = reversedShape[1]

            s += "\n"

            var arrayString = ""

            for i in 0 ... nElements {
                if i % lastDim == 0, i > 0 {
                    s += "    [\(arrayString)],\n"
                    arrayString = ""
                }

                if i % (lastDim * secondLastDim) == 0, i > 0, i < nElements {
                    s += "\n"
                }

                if i < nElements {
                    arrayString += "\(dataValue(at: i))" + ((i + 1) % lastDim != 0 ? ", " : "")
                }
            }
            return s + String(repeating: "]", count: max(shape.count - 1, 1)) + ")"
        }
    }
}