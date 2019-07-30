import Foundation

protocol DimensionProtocol {
    var count: Int { get }
    var realCount: Int { get }
    func realIndex(of: Int) -> Int
    func indexIterator() -> AnyIterator<Int>
}

extension DimensionProtocol {
    func fullSequence(cycleRepetitions: Int, elementRepetitions: Int) -> AnySequence<Int> {
        return AnySequence { () -> AnyIterator<Int> in
            var cycle = 0
            var repetition = 0
            var iterator = self.indexIterator()
            var current = iterator.next()

            return AnyIterator { () -> Int? in

                if repetition == elementRepetitions {
                    current = iterator.next()
                    repetition = 0
                }

                repetition += 1

                if current == nil {
                    cycle += 1

                    if cycle == cycleRepetitions {
                        return nil
                    }

                    iterator = self.indexIterator()
                    current = iterator.next()
                }

                return current
            }
        }
    }
}

struct ContiguousDimension: DimensionProtocol {
    let start: Int
    let end: Int

    var count: Int { end }
    var realCount: Int { end }
    func realIndex(of index: Int) -> Int { index }

    func indexIterator() -> AnyIterator<Int> {
        AnyIterator((start ..< end).makeIterator())
    }
}

struct Dimension: DimensionProtocol {
    let start: Int
    let end: Int
    let stride: Int
    let repetitions: Int

    var count: Int {
        if (start - end) % stride != 0 {
            return (1 + (end - start) / stride) * repetitions
        } else {
            return ((end - start) / stride) * repetitions
        }
    }

    var realCount: Int {
        return end
    }

    func realIndex(of index: Int) -> Int {
        precondition(index < count, "Index out bounds, index: \(index), count: \(count)")

        return start + (index / repetitions) * stride
    }

    func indexIterator() -> AnySequence<Int>.Iterator {
        let sequence = AnySequence { () -> AnyIterator<Int> in
            var current = self.start
            var repetition = 0

            return AnyIterator { () -> Int? in
                defer {
                    repetition += 1

                    if repetition == self.repetitions {
                        current += self.stride
                        repetition = 0
                    }
                }

                if current < self.end {
                    return current
                } else {
                    return nil
                }
            }
        }

        return sequence.makeIterator()
    }
}