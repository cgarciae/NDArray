
struct A: Differentiable {
    var a: NDArray<Float>
    var b: NDArray<Float>
}

let a = A(a: NDArray(1), b: NDArray(4))

// let da = a.gradient { a in
//     (a.a * a.b)[0]
// }

// print(da)