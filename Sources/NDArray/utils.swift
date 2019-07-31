
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