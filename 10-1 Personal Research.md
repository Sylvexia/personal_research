Task:
1. func to llvm dialect to llvm ir and make it executable with libm
	1. onnx-mlir-opt --convert-xxx-pass
		1. Need to see the real mlir conversion
2. Use polygeist to lower the c function code.
	1. Does not need c++, c++ requires name mangling, c does not.
3. What is extern C? or how to create a c wrapper around C++?
	1. See universal library
4. How to get the symbol string of a library?
	1. c should be the same
		1. need verify

6. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U