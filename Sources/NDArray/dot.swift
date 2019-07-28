
import func CBlas.cblas_sdot

public extension NDArray where Scalar == Float {
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