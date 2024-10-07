# Quantize inspiration
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

# KrnlGlobalOp

- Tablegen declaration:
	```cpp
	def KrnlGlobalOp : Op<Krnl_Dialect, "global", [Pure, MemRefsNormalizable]> {
	  let arguments = (ins AnyAttr:$shape,
	    StrAttr:$name, OptionalAttr<AnyAttr>:$value, OptionalAttr<I64Attr>:$offset,
	    OptionalAttr<I64Attr>:$alignment);
	  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);
	}
	```
- MLIR Example:
	`%1 = "krnl.global"() {name = "constant_2", shape = [32, 1, 3, 3], value = dense<"0x2F9C9F...> : tensor<32x1x3x3xf32>} : () -> memref<32x1x3x3xf32>`
- For our lowering, we might need to care the following attribute
	- value
		- scalar/dense attribute
	- offset
		- memory offset from the base address of the global buffer
	- alignment
		- memory address of the data should be a multiple of 8 bytes
		- `memref.global`
			- [nobody needs it](https://discourse.llvm.org/t/alignment-on-memref-global/3381)

# Two's complement
- https://stackoverflow.com/questions/25754082/how-to-take-twos-complement-of-a-byte-in-c

https://www.youtube.com/watch?v=UP-LBRbvI_U