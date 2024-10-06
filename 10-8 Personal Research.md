Task:
5. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

llvm
quantizeFloatToInt

tpu-mlir
```cpp
quant::UniformQuantizedType getUniformQuantizedType(Value v) {
  return v.getType()
      .cast<RankedTensorType>()
      .getElementType()
      .cast<quant::UniformQuantizedType>();
}
```

# Convert all Arith::Const F32 to UINT32

https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/

-debug to list the pattern convert

# `Arith` to Posit Function Call lowering experiment:

- Goal: Lowering `arith` Dialect to Posit Function Call
- Currently we only "proof of concept" the add and const operator for prototype
	- Insight:
		- Convert `f32` to `i32` for all `arith` float operation.
		- For binary operation like `addf`, `mulf`, ...
			- Map it to function symbol. e.g. `posit8es0_add`
			- Need the declaration and call.
		- For Constant
			- Extract raw bit and make the raw bit fit into posit format.
	- For Const Operation, we only implement proof of concept:
		- We can extract raw bit of float from `APFloat` Class
		- We simply Shift the raw bit to left for proof of concept
			- Indicate that we can modify the number without issue.
			- In short term goal, we would like to extract float raw bit and convert such that comply with "posit standard"
				- Actually when interfacing the universal library, need to apply 2's complement to raw bit when negative.
		- code:
			```cpp
			APFloat apFloat = floatAttr.getValue();
			uint64_t floatBits = apFloat.bitcastToAPInt().getZExtValue();
			int64_t intValue = static_cast<int64_t>(floatBits >> 1);
			```
	- For Add Operation:
		- Generating the symbol name by `opName`, `nbits`, `es_val`
		- API: Just register the following to pass to add Add operation lowering.
			- `populateConvertArithAddToPositFuncPattern(patterns, typeConverter, "add", 8, 0);`
			- Which would create `posit8es0_add` declaration and call.
	- If we were to implement operations should be like wise.
	- For return type materialization issue from last report
		- Currently it's a workaround I found in this tutorial:
			- Using existing conversion pattern to do the conversion.
			- https://www.jeremykun.com/2023/10/23/mlir-dialect-conversion/
			- Good tutorial by the way.
		- There's other tutorial doing the `returnOp` lowering.
		- I should see other's real codebase for reference.
	- Experiment result:
		- Test Case:
			```cpp
			func.func @test_arith(%arg0 : f32, %arg1 : f32) {
			  %0 = arith.constant 1.0 : f32
			  %1 = arith.constant 2.0 : f32
			  %2 = arith.addf %arg0, %arg1 : f32
			  return
			}
			
			func.func @test_arith_prop(%arg0 : f32, %arg1 : f32) {
			  %0 = arith.constant 1.0 : f32
			  %1 = arith.constant 2.0 : f32
			  %2 = arith.addf %0, %1 : f32
			  return
			}
			
			func.func @test_arith_const(%arg0 : f32, %arg1 : f32) {
			  %0 = arith.constant 1.0 : f32
			  %1 = arith.constant 2.0 : f32
			  %2 = arith.addf %arg0, %1 : f32
			  return
			}
			
			func.func @test_const(%float1: f32, %float2: f32) {
			  %0 = arith.constant 1.69 : f32
			  %1 = arith.constant 3.14 : f32
			  return
			}
			
			func.func @test_const_return(%float1: f32, %float2: f32) -> (f32, f32) {
			  %0 = arith.constant 2.68 : f32
			  %1 = arith.constant 6.9 : f32
			  return %0, %1 : f32, f32
			}
			```
		- Command: 
			- `./onnx-mlir-opt /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir --convert-arith-to-posit-func`
		- Result:
			```cpp
			module {
			  func.func private @posit8es0_add(i32, i32) -> i32 attributes {llvm.readnone}
			  func.func @test_arith(%arg0: i32, %arg1: i32) {
			    %c532676608_i32 = arith.constant 532676608 : i32
			    %c536870912_i32 = arith.constant 536870912 : i32
			    %0 = call @posit8es0_add(%arg0, %arg1) : (i32, i32) -> i32                  return
			  }
			  func.func @test_arith_prop(%arg0: i32, %arg1: i32) {
			    %c532676608_i32 = arith.constant 532676608 : i32
			    %c536870912_i32 = arith.constant 536870912 : i32
			    %c538968064_i32 = arith.constant 538968064 : i32
			    return
			  }
			  func.func @test_arith_const(%arg0: i32, %arg1: i32) {
			    %c532676608_i32 = arith.constant 532676608 : i32
			    %c536870912_i32 = arith.constant 536870912 : i32
			    %0 = call @posit8es0_add(%arg0, %c536870912_i32) : (i32, i32) -> i32
			    return
			  }
			  func.func @test_const(%arg0: i32, %arg1: i32) {
			    %c535570678_i32 = arith.constant 535570678 : i32
			    %c539261665_i32 = arith.constant 539261665 : i32
			    return
			  }
			  func.func @test_const_return(%arg0: i32, %arg1: i32) -> (i32, i32) {
			    %c538296975_i32 = arith.constant 538296975 : i32
			    %c544106086_i32 = arith.constant 544106086 : i32
			    return %c538296975_i32, %c544106086_i32 : i32, i32
			  }
			}
			```
		- Observation
			- If both add input argument is constant, it would calculate for you and reduce the result as constant, hence there's no posit function call.
				- Would this cause the result be incorrect for our use case?
			- Multiple calls to a same function would have only one function declaration.
	- Mistakes and tips:
		- When converting the operation, always remember of convert based on its operand and result instead of creating new.
			- with debug, it might shows the operand link breaks
		- Remember to register the legal/illegal ops/dialect
			- If has no materialization error but it still not converting, chances are you forget it.
		- add `-debug` helps with log out the conversion process.
- Future Works:
	- Refactor the current implementation.
	- Model constant is not all in `arith` const, mostly on `krnl.global`:
		- This is assume that we lower before the `llvm-mlir`
			- LLVM MLIR dialect does not have `arith` dialect.
		- We need to lower custom `krnl.global` ourself.
		- Example:
			- `%1 = "krnl.global"() {name = "constant_2", shape = [32, 1, 3, 3], value = dense<"0x2F9C9F...> : tensor<32x1x3x3xf32>} : () -> memref<32x1x3x3xf32>`
			- Command:
				- `./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
	- Implement other operations in `mlir` for `MNIST` model
		- `addf`, `cmpf`, `constant`, `mulf`, `select`
			- `select`: 
				- `%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32`
				- Like ternary operator, if true or false is float we need to convert it.
	- Proof of correctness?
		- `mlir-cpu-runner` testcase?
			- reference command for pass pipeline:
				```cpp
				// RUN: mlir-opt %s \
				// RUN:   -pass-pipeline="builtin.module( \
				// RUN:      convert-math-to-funcs{convert-ctlz}, \
				// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
				// RUN:      convert-func-to-llvm, \
				// RUN:      convert-cf-to-llvm, \
				// RUN:      reconcile-unrealized-casts)" \
				// RUN: | mlir-cpu-runner -e test_7i32_to_29 -entry-point-result=i32 > %t
				// RUN: FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29 < %t
				```
	- Continue the works of universal library wrapper
	- See how the quantize going in `tensorflow`.
		- By seeing this we can be sure of the real implementation of type conversion and value mapping.
	- See how the `@run_main_graph` for entry point of a model get implemented.

