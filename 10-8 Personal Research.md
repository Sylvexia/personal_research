Task:
5. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

onnx @run_main_graph

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

AddLowering::LoweringINT8

for operator respectively?

get values from dense element
https://discourse.llvm.org/t/using-mlir-getvalues-with-f16/3953/5

opRewritePattern v.s opConversionPattern
# Convert all Arith::Const F32 to UINT32

https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/

-debug to list the pattern convert

https://mlir.llvm.org/docs/DialectConversion/#type-conversion

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
	- For return type issue like 
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
			- Multiple calls to a same function would have only one function declaration
- Future Works:
	- Implement other operations in `mlir` for `MNIST` model and make it runnable
		- `addf`, `cmpf`, `constant`, `mulf`, `select`
			- `select`: 
				- `%x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32`
				- Like ternary operator, if true or false is float we need to convert it.

