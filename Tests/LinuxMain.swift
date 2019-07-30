import XCTest

import NDArrayTests

var tests = [XCTestCaseEntry]()
tests += NDArrayTests.allTests()
XCTMain(tests)