- `Tensorflow` inspiration:
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/tests/quantize.mlir
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/stablehlo/BUILD
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/tensorflow/tests/add_quantization_unit_loc.mlir
- Keywords
- llvm
	- quantizeFloatToInt
- tpu-mlir
	```cpp
	quant::UniformQuantizedType getUniformQuantizedType(Value v) {
	  return v.getType()
	      .cast<RankedTensorType>()
	      .getElementType()
	      .cast<quant::UniformQuantizedType>();
	}
	```

https://www.youtube.com/watch?v=UP-LBRbvI_U