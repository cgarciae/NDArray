import Foundation

public func indexSequence(range: Range<Int>, shape: [Int]) -> AnySequence<(linearIndex: Int, rectangularIndex: UnsafeMutableBufferPointer<Int>)> {
    AnySequence { () -> AnyIterator<(linearIndex: Int, rectangularIndex: UnsafeMutableBufferPointer<Int>)> in
        let arrayShape = shape

        var iterator = range.makeIterator()
        let dimensionStrides = getDimensionStrides(of: arrayShape)

        let rectangularIndex = UnsafeMutableBufferPointer<Int>.allocate(capacity: arrayShape.count)
        rectangularIndex.initialize(repeating: 0)
        let shape = UnsafeMutableBufferPointer<Int>.allocate(capacity: arrayShape.count)
        _ = shape.initialize(from: arrayShape)

        var first = true
        return AnyIterator { () -> (linearIndex: Int, rectangularIndex: UnsafeMutableBufferPointer<Int>)? in
            guard let current = iterator.next() else {
                shape.baseAddress!.deinitialize(count: shape.count)
                shape.deallocate()

                rectangularIndex.baseAddress!.deinitialize(count: rectangularIndex.count)
                rectangularIndex.deallocate()

                return nil
            }

            if first {
                first = false
                var remainder = current

                for i in 0 ..< shape.count {
                    if shape[i] > 1 {
                        let index: Int
                        (index, remainder) = remainder.quotientAndRemainder(dividingBy: dimensionStrides[i])

                        rectangularIndex[i] = index
                    }
                }

                return (
                    linearIndex: current,
                    rectangularIndex: rectangularIndex
                )
            }

            var pos = shape.count - 1

            while pos >= 0 {
                let nextValue = rectangularIndex[pos] + 1
                if nextValue % shape[pos] == 0 {
                    rectangularIndex[pos] = 0
                    pos -= 1
                } else {
                    rectangularIndex[pos] = nextValue
                    break
                }
            }

            return (
                linearIndex: current,
                rectangularIndex: rectangularIndex
            )
        }
    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< $0 + size])
        }
    }
}

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

@inlinable
public func parFor<S: Sequence, E>(
    _ range: Range<Int>,
    workers: Int = CPU_COUNT,
    rangeMap: @escaping (Range<Int>) -> S,
    body: @escaping (E) -> Void
) where S.Element == E {
    let group = DispatchGroup()
    let nElements = range.count

    for range in splitRanges(total: nElements, splits: workers) {
        group.enter()

        DISPATCH.async {
            for element in rangeMap(range) {
                body(element)
            }
            group.leave()
        }
    }

    group.wait()
}

@inlinable
public func parFor(
    _ range: Range<Int>,
    workers: Int = CPU_COUNT,
    body: @escaping (Int) -> Void
) {
    parFor(range, workers: workers, rangeMap: { x -> Range<Int> in x }, body: body)
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

extension Sequence where Element: Numeric {
    @inlinable
    public func sum() -> Element {
        reduce(0, +)
    }

    @inlinable
    public func product() -> Element {
        reduce(1, *)
    }
}

extension Sequence where Element == Bool {
    @inlinable
    public func all() -> Element {
        reduce(true) { $0 && $1 }
    }

    @inlinable
    public func product() -> Element {
        reduce(false) { $0 || $1 }
    }
}