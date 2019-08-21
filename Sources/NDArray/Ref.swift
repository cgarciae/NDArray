

public final class Ref<A>: CustomStringConvertible {
    @usableFromInline var value: A

    @usableFromInline init(_ value: A) {
        self.value = value
    }

    public var description: String { "\(value)" }
}