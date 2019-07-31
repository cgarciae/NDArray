
public func flattenArrays<A>(_ array: [Any]) -> (array: [A], shape: [Int]) {
    if array.count == 0 {
        return (
            array: [],
            shape: []
        )
    } else {
        var shape: [Int] = [array.count]
        return flattenArrays(array: array, shape: &shape)
    }
}

func flattenArrays<A>(array: [Any], shape: inout [Int]) -> (array: [A], shape: [Int]) {
    if array[0] is A {
        return (
            array: array as! [A],
            shape: shape
        )

    } else {
        let array = array as! [[Any]]
        shape.append(array[0].count)
        return flattenArrays(
            array: array.flatMap { $0 },
            shape: &shape
        )
    }
}

@inlinable
public func splitRanges(total: Int, splits: Int) -> [Range<Int>] {
    let total = Float(total)
    let points = Set(
        (0 ... splits).map { (s: Int) -> Int in
            let percent = Float(s) / Float(splits)
            return Int(total * percent)
        }
    )

    let orderedPoints = Array(points).sorted()
    var ranges = [Range<Int>]()

    for i in 0 ..< orderedPoints.count - 1 {
        ranges.append(orderedPoints[i] ..< orderedPoints[i + 1])
    }

    return ranges
}

@inlinable
public func getDimensionStrides(of shape: [Int]) -> [Int] {
    shape.reversed().scan(*).reversed().dropFirst() + [1]
}

extension Sequence {
    @inlinable
    func scan(initial: Element? = nil, _ f: @escaping (Element, Element) -> Element) -> AnySequence<Element> {
        AnySequence<Element> { () -> AnyIterator<Element> in
            var iterator = self.makeIterator()
            var acc = initial ?? iterator.next()
            var mayberNext = iterator.next()

            return AnyIterator { () -> Element? in
                if let next = mayberNext {
                    defer {
                        mayberNext = iterator.next()
                        acc = f(acc!, next)
                    }

                    return acc
                } else {
                    defer {
                        acc = nil
                    }
                    return acc
                }
            }
        }
    }
}