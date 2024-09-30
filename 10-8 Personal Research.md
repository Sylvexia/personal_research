Task:
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