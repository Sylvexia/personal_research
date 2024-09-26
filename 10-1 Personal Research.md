Task:
1. func to llvm dialect to llvm ir and make it executable with libm
	1. onnx-mlir-opt --convert-xxx-pass
		1. Need to see the real mlir conversion
3. What is extern C? or how to create a c wrapper around C++?
	1. See universal library
4. How to get the symbol string of a library?
	1. c should be the same

6. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

# Summary

- For llvm dialect, take invoking `exp` for example:
	- If we can create the declaration, invoke and compile with -lm:
		- Declaration: `llvm.func @exp(%arg0: f64) -> f64`
		- Invoke: `%4 = llvm.call @exp(%0) : (f64) -> f64`
	- With:
		- `bin/mlir-translate -mlir-to-llvmir test_mod.mlir -o test_mod.ll`
		- `llc --filetype=obj --relocation-model=pic test_mod.ll -o test_mod.o`
		- `bin/clang -lm test_mod.o -o test_mod.exe`