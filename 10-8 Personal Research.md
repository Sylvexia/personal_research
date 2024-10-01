Task:

Task:
5. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

what is casting in `mlir`?
- When getting the operation
	- For example, `AddOp` is derived from operation
	- We cast the operation to decide if it's `AddOp`
		- cast<>

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

Exp: