# NDArray

NDArray is a multidimensional array library written in Swift that aims to become the equivalent of `numpy` in Swift's emerging data science ecosystem. This project is in a very early stage and has a long but exciting road ahead!

## Goals

1. Have an efficient multidimensional array interface with common things like indexing, slicing, broadcasting, etc. 
2. Create specialized implementations of linear algebra operations for NDArrays containing numeric types using libraries like BLAS and LAPACK.
3. Make `NDArray` and its operations `differentiable` so its usable along with Swift for TensorFlow.

## Installation
You can install it via SwiftPM via:
```swift
.package(url: "https://github.com/cgarciae/NDArray", from: "0.0.2")
```
It might work on other compatible package managers. This package is only tested in Swift 5.1, compatibility with previous version is not guaranteed.  Although specified in the `Package.swift` file you might also need to run these commands to setup your environment.

#### Ubuntu
```bash
sudo apt-get install gfortran liblapack3 liblapacke liblapacke-dev libopenblas-base libopenblas-dev
```
#### OSX
```bash
brew install homebrew/dupes/lapack homebrew/science/openblas
```

## Example
`NDArray` is a generic container type just like `Array` with the difference that its multidimensional. If its elements conform to certain protocols then certain methods and operators like `+`, `-`, `*`, etc, can be used to efficiently perform computations of the whole collection.
```swift
import NDArray

let a = NDArray<Int>([
    [1, 2, 3],
    [4, 5, 6],
])
let b = NDArray<Int>([
    [7, 8, 9],
    [10, 11, 12],
])

print((a + b) * a)
/*
NDArray<Int>[2, 3]([
    [8, 20, 36],
    [56, 80, 108],
])
*/
```
Here we see that the outcome of `(a + b) * a` is also and `NDArray` of `Int` with shape `[2, 3]`. To use operators like `+` and `*` with NDArrays containing your custom types you just have to make them conform to the proper protocols. For example:
```swift
import NDArray

struct Point: AdditiveArithmetic {
    let x: Float
    let y: Float
    ...
}

let a = NDArray<Point>([Point(x: 1, y: 2), Point(x: 2, y: 3)])
let b = NDArray<Point>([Point(x: 4, y: 5), Point(x: 6, y: 7)])

print(a + b)
/*
NDArray<Point>[2]([Point(x: 5.0, y: 7.0), Point(x: 8.0, y: 10.0)])
*/
```
You can also apply generic transformations over the data, the previous could have been written as:
```swift
elementwise(a, b, apply: +)
// or
elementwise(a, b) { $0 + $1 }
```
For heavy computation you can use the parallelized version:
```swift
elementwiseInParallel(a, b) {
    // code
    return c
}
```
In the future `NDArray` should be able to estimate the best strategy (serial/parallelized) based on the type and size of the data.

## Goals Discussion
Each of NDArray's goals builds upon the previous and users should be able to opt-in for the features they need or can have access to in their environment.

#### Basic API
The first goal is the definition of the library's basic API using pure Swift with no extra optimization or differentiable capabilities. iOS/OSX developers should be able to use the basic API without additional setup. It will also be important to keeping the NDArray's API in close coordination with Swift for TensorFlow's Tensor API to promote knowledge reuse and free documentation if possible.

#### Linear Algebra Optimization
The second goal is what you would expect from a HPC numeric library and thanks to Swift's capability of extending types and specializing functions it seems very plausible to implement this as an addon without loosing performance.

#### Automatic Differentiation
The third an obvious must have, Swift for TensorFlow's compiler with automatic differentiation is arguably the future of ML and we should use it.

## Roadmap
##### 0.1
- [x] Operators: `+`, `-`, `*`, `\`
- [x] Indexing
- [x] Slicing
- [x] Pretty Print
- [ ] Broadcasting (coming soon)
- [ ] Assignment (coming soon)
- [ ] More operators
- [ ] API + Codebase cleanup
- [ ] Documentation
##### 0.2
- [x] Link BLAS and LAPACK
- [ ] Specialize operators using BLAS and LAPACK
- [ ] `dot` product, and others
Initial 
##### 0.3
- [ ] Differentiable conformance

## Meta
Cristian Garcia – cgarcia.e88@gmail.com

Distributed under the MIT license. See LICENSE for more information.

