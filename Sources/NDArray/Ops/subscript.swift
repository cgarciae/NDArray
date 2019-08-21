import Foundation

public let all = ArrayRange.all
public let newAxis = ArrayRange.newAxis
public let ellipsis = ArrayRange.ellipsis
public let squeezeAxis = ArrayRange.squeezeAxis

// experimental alias
public let rest = ArrayRange.ellipsis
public let new = ArrayRange.newAxis
public let squeeze = ArrayRange.squeezeAxis

extension NDArray {
    @inlinable
    public subscript(_ ranges: ArrayExpression...) -> NDArray {
        get {
            self[r: ranges.map { $0.arrayRange }]
        }
        mutating set(ndarray) {
            self[r: ranges.map { $0.arrayRange }] = ndarray
        }
    }

    @inlinable
    public subscript(r ranges: [ArrayRange]) -> NDArray {
        get {
            subscript_get(ranges)
        }

        mutating set(ndarray) {
            self = subscript_set(ranges, ndarray)
        }
    }
}

public protocol ArrayExpression {
    @inlinable
    var arrayRange: ArrayRange { get }
}

extension Array: ArrayExpression {
    public var arrayRange: ArrayRange {
        if self is [Int] {
            return .filter(self as! [Int])
        } else if self is [Bool] {
            let array = self as! [Bool]
            return .filter(array.enumerated().filter { $0.1 }.map { $0.0 })
        } else {
            fatalError("Type \(Element.self) not supported")
        }
    }
}

public enum ArrayRange: ArrayExpression {
    case ellipsis
    case newAxis
    case squeezeAxis
    case all
    case index(Int)
    case filter([Int])
    case slice(start: Int? = nil, end: Int? = nil, stride: Int = 1)

    public var arrayRange: ArrayRange { self }
}

public func isEllipsis(_ range: ArrayRange) -> Bool {
    switch range {
    case .ellipsis:
        return true
    default:
        return false
    }
}

extension Int: ArrayExpression {
    public var arrayRange: ArrayRange { .index(self) }
}

extension Range: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .slice(start: lowerBound, end: upperBound) }
    public func stride(_ stride: Int) -> ArrayRange {
        .slice(start: lowerBound, end: upperBound, stride: stride)
    }
}

extension ClosedRange: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .slice(start: lowerBound, end: upperBound + 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .slice(start: lowerBound, end: upperBound + 1, stride: stride)
    }
}

extension PartialRangeFrom: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .slice(start: lowerBound) }
    public func stride(_ stride: Int) -> ArrayRange {
        .slice(start: lowerBound, stride: stride)
    }
}

extension PartialRangeUpTo: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .slice(end: upperBound) }
    public func stride(_ stride: Int) -> ArrayRange {
        .slice(end: upperBound, stride: stride)
    }
}

extension PartialRangeThrough: ArrayExpression where Bound == Int {
    public var arrayRange: ArrayRange { .slice(end: upperBound + 1) }
    public func stride(_ stride: Int) -> ArrayRange {
        .slice(end: upperBound + 1, stride: stride)
    }
}

public struct Slice: ArrayExpression {
    let start: Int?
    let end: Int?

    internal init(start: Int? = nil, end: Int? = nil) {
        self.start = start
        self.end = end
    }

    public var arrayRange: ArrayRange {
        .slice(start: start, end: end)
    }
}

public struct StridedSlice : ArrayExpression {
    let start: Int?
    let end: Int?
    let stride: Int

    internal init(start: Int? = nil, end: Int? = nil, stride: Int = 1) {
        self.start = start
        self.end = end
        self.stride = stride
    }

    public var arrayRange: ArrayRange {
        .slice(start: start, end: end, stride: stride)
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// ..
/////////////////////////////////////////////////////////////////////////////////////////////

postfix operator ..
prefix operator ..
prefix operator ..-
infix operator ..: AdditionPrecedence
infix operator ..-: AdditionPrecedence
infix operator ....
prefix operator ....
prefix operator ....-

public extension Int {
    static postfix func .. (lhs: Int) -> Slice {
        Slice(start: lhs)
    }

    static prefix func .. (rhs: Int) -> Slice {
        Slice(end: rhs)
    }

    static prefix func ..- (rhs: Int) -> Slice {
        Slice(end: -rhs)
    }

    static func .. (lhs: Int, rhs: Int) -> Slice {
        Slice(start: lhs, end: rhs)
    }

    static func .. (lhs: Slice, rhs: Int) -> StridedSlice {
        StridedSlice(start: lhs.start, end: lhs.end, stride: rhs)
    }

    static func ..- (lhs: Int, rhs: Int) -> Slice {
        Slice(start: lhs, end: -rhs)
    }

    static func ..- (lhs: Slice, rhs: Int) -> StridedSlice {
        StridedSlice(start: lhs.start, end: lhs.end, stride: -rhs)
    }

    static prefix func .... (rhs: Int) -> StridedSlice {
        StridedSlice(stride: rhs)
    }

    static prefix func ....- (rhs: Int) -> StridedSlice {
        StridedSlice(stride: -rhs)
    }

    static func .... (lhs: Int, rhs: Int) -> StridedSlice {
        StridedSlice(start: lhs, stride: rhs)
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// |>
/////////////////////////////////////////////////////////////////////////////////////////////

// postfix operator |>
// prefix operator |>
// prefix operator |>-
// infix operator |>: AdditionPrecedence
// infix operator ||>
// prefix operator ||>
// prefix operator ||>-

// public extension Int {
//     static postfix func |> (lhs: Int) -> Slice {
//         Slice(start: lhs)
//     }

//     static prefix func |> (rhs: Int) -> Slice {
//         Slice(end: rhs)
//     }

//     static prefix func |>- (rhs: Int) -> Slice {
//         Slice(end: -rhs)
//     }

//     static func |> (lhs: Int, rhs: Int) -> Slice {
//         Slice(start: lhs, end: rhs)
//     }

//     static func |> (lhs: Slice, rhs: Int) -> Slice {
//         Slice(start: lhs.start, end: lhs.end, stride: rhs)
//     }

//     static func |> (lhs: Int, rhs: Slice) -> Slice {
//         Slice(start: lhs, end: rhs.start, stride: rhs.end!)
//     }

//     static prefix func ||> (rhs: Int) -> Slice {
//         Slice(stride: rhs)
//     }

//     static prefix func ||>- (rhs: Int) -> Slice {
//         Slice(stride: -rhs)
//     }

//     static func ||> (lhs: Int, rhs: Int) -> Slice {
//         Slice(start: lhs, stride: rhs)
//     }
// }