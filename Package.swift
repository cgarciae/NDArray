// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NDArray",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "NDArray",
            targets: ["NDArray"]
        ),
        .library(
            name: "Debug",
            targets: ["Debug"]
        ),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: "../NDArray-clibs", from: "2.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        // .systemLibrary(
        //     name: "CBlas",
        //     path: "Sources/C/CBlas",
        //     pkgConfig: "blas",
        //     providers: [
        //         .apt(["gfortran", "liblapack3", "liblapacke", "liblapacke-dev", "libopenblas-base", "libopenblas-dev"]),
        //         .brew(["homebrew/dupes/lapack", "homebrew/science/openblas"]),
        //     ]
        // ),
        // .systemLibrary(
        //     name: "CLapack",
        //     path: "Sources/C/CLapack",
        //     pkgConfig: "lapack",
        //     providers: [
        //         .apt(["gfortran", "liblapack3", "liblapacke", "liblapacke-dev", "libopenblas-base", "libopenblas-dev"]),
        //         .brew(["homebrew/dupes/lapack", "homebrew/science/openblas"]),
        //     ]
        // ),
        .target(
            name: "NDArray",
            dependencies: []
            // dependencies: ["CBlas", "CLapack"]
        ),
        .target(
            name: "Debug",
            dependencies: ["NDArray"]
        ),
        .testTarget(
            name: "NDArrayTests",
            dependencies: ["NDArray"]
        ),
    ]
)