import XCTest

#if !canImport(ObjectiveC)
    public func allTests() -> [XCTestCaseEntry] {
        return [
            testCase(NDArrayTests.allTests),
            testCase(DimensionTests.allTests),
        ]
    }
#endif