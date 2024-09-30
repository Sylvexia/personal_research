Task:

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