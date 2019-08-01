# NDArray

NDArray is a multidimensional array library written in Swift that aims to become the equivalent of `numpy` in Swift's emerging data science ecosystem.

NDArray is in a very early stage and has a long but exciting rode ahead, contributions are welcome!

## Installation


## Example

## Goals
NDArray has several goals, each builds upon the previous but users should be able to opt-in for the features depending on their environment:

1. Have a proper multidimensional array interface with common things like indexing, slicing, broadcasting, etc. 
2. Create specialized implementations of linear algebra operations for NDArrays containing numeric types using libraries like BLAS and LAPACK.
3. Make `NDArray` and its operations differentiable so its usable for machine learning applications that need automatic differentiation.

The first goal is the definition of the library's basic API using pure Swift with no extra optimization or differentiable capabilities. iOS/OSX developers should be able to use the basic API without additional setup. It will also be important to keeping the NDArray's API in close coordination with Swift for TensorFlow's Tensor API to promote knowledge reuse and free documentation if possible.

The second goal is what you would expect from a HPC numeric library and thanks to Swift's capability of extending types and specializing functions it seems very plausible to implement this as an addon without loosing performance.

The third an obvious (must have?) extension that leverages the Swift for TensorFlow's compiler. 

## Architecture

## Roadmap


## Meta
Cristian Garcia – cgarcia.e88@gmail.com

Distributed under the MIT license. See LICENSE for more information.

