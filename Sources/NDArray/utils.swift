

// Neither BLAS or LAPACK have element-wise multiplication, MKL has it but it would make
// this library Intel-dependent. Since its so simple a parrallelized element-wise generic operation
// could be implemented in pure Swift using GCD.
func elementWise<T>(
    between arrayA: [T],
    and arrayB: [T],
    apply f: (T, T) -> T
) -> [T] {
    precondition(arrayA.count == arrayB.count)

    var arrayC = Array(arrayA)

    arrayC.withUnsafeMutableBufferPointer { arrayC in
        arrayA.withUnsafeBufferPointer { arrayA in
            arrayB.withUnsafeBufferPointer { arrayB in
                for i in 0 ..< arrayA.count {
                    arrayC[i] = f(arrayA[i], arrayB[i])
                }
            }
        }
    }

    return arrayC
}