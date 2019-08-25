import Foundation

public protocol Divisible: Numeric {
    static func / (lhs: Self, rhs: Self) -> Self
    static func /= (lhs: inout Self, rhs: Self)
}

extension Divisible {
    public static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
    }
}

extension Int: Divisible {}
extension Int8: Divisible {}
extension Int16: Divisible {}
extension Int32: Divisible {}
extension Int64: Divisible {}

extension UInt: Divisible {}
extension UInt8: Divisible {}
extension UInt16: Divisible {}
extension UInt32: Divisible {}
extension UInt64: Divisible {}

extension Double: Divisible {}
extension Float: Divisible {}
extension Float80: Divisible {}
extension CGFloat: Divisible {}