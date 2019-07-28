import func CBlas.cblas_saxpy
import func CBlas.cblas_sdot
// import CLapack

// let x = cblas_srot(value, value, value, value, value, value, value)

struct NDArray<Element> {
    var data: [Element]
    let shape: [Int]
}

extension NDArray: Differentiable where Element: Differentiable {
    typealias TangentVector = NDArray
    typealias AllDifferentiableVariables = NDArray
}

func * (_ array: NDArray<Float>, _ scalar: Float) -> NDArray<Float> {
    var output = [Float](repeating: 0, count: array.data.count)
    cblas_saxpy(Int32(array.data.count), scalar, array.data, 1, &output, 1)

    return NDArray(data: output, shape: array.shape)
}

extension NDArray where Element == Float {
    func dot(_ array: NDArray<Float>) -> Float {
        return cblas_sdot(Int32(data.count), data, 1, array.data, 1)
    }

    // @differentiating(dot, wrt: (self, array))
    // func dotGrad(_ array: NDArray<Float>) -> (value: Float, pullback: (Float) -> (NDArray<Float>, NDArray<Float>)) {
    //     let y = dot(array)

    //     return (value: y, pullback: { dy in
    //         let dself = self * dy
    //         let darray = array * dy

    //         return (dself, darray)
    //     })
    // }
}

let a = NDArray<Float>(data: [1, 1, 1], shape: [3])
let b = NDArray<Float>(data: [2, 2, 2], shape: [3])

print(a.dot(b))