# Task
- Validate the posit conversion at `mlir` side.
- Revise the posit converter implementation.
- Nar handling in universal 


# Quantize inspiration
- `Tensorflow` inspiration:
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/tests/quantize.mlir
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/stablehlo/BUILD
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/tensorflow/tests/add_quantization_unit_loc.mlir
- Keywords
	- `llvm`
		- quantizeFloatToInt
	- `tpu-mlir`
		```cpp
		quant::UniformQuantizedType getUniformQuantizedType(Value v) {
		  return v.getType()
		      .cast<RankedTensorType>()
		      .getElementType()
		      .cast<quant::UniformQuantizedType>();
		}
		```

# MLIR structure

The following MLIR is before lower to `llvm dialect`
- `--EmitMLIR - Lower the input to MLIR built-in transformation dialect.`

```cpp
%reinterpret_cast = memref.reinterpret_cast %alloc_5 to offset: [0], sizes: [1, 3136], strides: [3136, 1] : memref<1x64x7x7xf32> to memref<1x3136xf32>
    %alloc_7 = memref.alloc() {alignment = 128 : i64} : memref<1x128xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        %alloca_10 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_10[] : memref<f32>
        affine.for %arg3 = 0 to 3136 {
          %11 = affine.load %reinterpret_cast[%arg1, %arg3] : memref<1x3136xf32>
          %12 = affine.load %5[%arg2, %arg3] : memref<128x3136xf32>
          %13 = arith.mulf %11, %12 : f32
          %14 = affine.load %alloca_10[] : memref<f32>
          %15 = arith.addf %13, %14 : f32
          affine.store %15, %alloca_10[] : memref<f32>
        }
        %8 = affine.load %alloca_10[] : memref<f32>
        %9 = affine.load %4[%arg2] : memref<128xf32>
        %10 = arith.addf %8, %9 : f32
        affine.store %10, %alloc_7[%arg1, %arg2] : memref<1x128xf32>
      }
    }
```
# `KrnlGlobalOp`

- `Tablegen` declaration:
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
- For our `krnl.global` lowering, we might need to care the following attribute
	- value
		- scalar/dense attribute
	- offset
		- memory offset from the base address of the global buffer
		- `memref.reinterprete_cast`:
			- https://discourse.llvm.org/t/question-about-memref-reinterpret-casts-offset/76082
	- alignment
		- memory address of the data should be a multiple of 8 bytes
		- `memref.global`
			- [nobody needs it](https://discourse.llvm.org/t/alignment-on-memref-global/3381)
	- Probably we don't need to touch offset and alignment.

# Two's complement
- In universal library, the negative is 2's complement for bit except the sign bit compare to standard
	- We must comply with posit standard at MLIR side, hence we need to deal with the issue in the posit wrapper.
	- Code snippet:
		```cpp
		template <size_t nbits, typename uType>
		void wrap(uType a, sw::universal::bitblock<nbits> &raw) {
		  for (size_t i = 0; i < nbits; i++) {
		    raw[i] = a & 1;
		    a >>= 1;
		  }
		  // if negative, two's complement except the sign bit
		  if (raw[nbits - 1]) {
		    sw::universal::bitblock<nbits - 1> remain;
		    for (size_t i = 0; i < nbits - 1; i++) {
		      remain[i] = raw[i];
		    }
		    remain = sw::universal::internal::twos_complement(remain);
		    for (size_t i = 0; i < nbits - 1; i++) {
		      raw[i] = remain[i];
		    }
		  }
		}
		```
- Test Case:
	- code snippet:
		```cpp
		uint8_t a = 0b11001000;//-1.25
	    uint8_t b = 0b01110101;//6.5
	    uint8_t c = posit8es0_add(a, b); //01110010
	    // 01110010 : 5
		```
	- 0b11101010;//-3.25
	- 0b01010000;//1.5
	- 11011000 : -1.75
- Reference for implementing 2's complement:
	- [casted with unsigned type, not used](https://stackoverflow.com/questions/25754082/how-to-take-twos-complement-of-a-byte-in-c)

https://www.youtube.com/watch?v=UP-LBRbvI_U