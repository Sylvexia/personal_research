# Task
- `MLIR`
	- Modify the `KrnlGlobalOp` value attribute.
	- Dispatch the type.
		- `./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=16' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir`
			- Posit value constant break
		- struct `FrontendToKrnlLoweringPass` can be referred
	- Writing a pass that convert all `f32` data type to say, `uint8`
	- Get to know the `private` and `readonly` attribute.
		- [ref](https://llvm.org/docs/LangRef.html)
		- `private`
			- Global values with “`private`” linkage are only directly accessible by objects in the current module. In particular, linking code into a module with a private global value may cause the private to be renamed as necessary to avoid collisions. Because the symbol is private to the module, all references can be updated. This doesn’t show up in any symbol table in the object file.
		- `readonly`
			- This attribute indicates that the function does not write through this pointer argument, even though it may write to the memory that the pointer points to.
			- If a function writes to a `readonly` pointer argument, the behavior is undefined.
	- what does `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` mean?
	- alignment is for SIMD?
	- `@run_main_graph`
		- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.
# Quantize inspiration
- `Tensorflow` inspiration:
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/tests/quantize.mlir
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/stablehlo/BUILD
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/quantization/tensorflow/tests/add_quantization_unit_loc.mlir
- Keywords
	- `llvm`
		- `quantizeFloatToInt`
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
	- command:
		- `./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
```cpp
// part of the code
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
- [`memref` reference](https://mlir.llvm.org/docs/Dialects/MemRef/)
- `reinterpret_cast`: takes an allocated memory of type `memref<1x64x7x7xf32>` and "views" it as a `memref<1x3136xf32>`
- `alloc`/`alloca`
	- `alloc` allocate memory on heap (verify)
		- `%alloc_7 = memref.alloc() {alignment = 128 : i64} : memref<1x128xf32>`
			- `alignment = 128` might infer `SIMD` or similar memory alignment requirements for performance.
	- `alloca` allocate memory on stack.
- `load`: 
	- `%9 = affine.load %4[%arg2] : memref<128xf32>`
		- means `reg%9` = `arr%4[arg2]`
		- `arr%4` might be something like `tensor_4[128]`
- `store`: 
	- `affine.store %10, %alloc_7[%arg1, %arg2] : memref<1x128xf32>` 
		- means array `alloc_7[arg1][arg2]` = `reg%10`
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
		- SIMD support?
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
	- How do I test it?
		- code snippet:
			```cpp
			// calculate with posit
	        uint8_t a = rand() % 256;
	        uint8_t b = rand() % 256;
	        uint8_t c = posit8es0_add(a, b); // our wrapper!
			// convert to posit to get double floating point
			// and calculate based on 
	        auto pa = get_posit<8, 0>(a);
	        auto pb = get_posit<8, 0>(b);
	        double da = static_cast<double>(pa);
	        double db = static_cast<double>(pb);
	        double dc = da + db;
			// convert to back to posit to get uint8_t
	        sw::universal::posit<8, 0> pc(dc);
	        uint8_t c_ref = get_uType<8, 0, uint8_t>(pc);
	        // compare c_ref and c
			```
	- output log:
		```bash
		//...
		PASS: a = 186 b = 1 c = 185 c_ref = 185
		PASS: a = 128 b = 87 c = 128 c_ref = 128
		Passed 255 tests
		```
- Verify the library name
	```bash
	$ nm libposit_c_api_custom.a | grep posit8
	0000000000000000 t _GLOBAL__sub_I_posit8es0_add
	0000000000000590 T posit8es0_add
	0000000000000990 T posit8es0_div
	0000000000001820 T posit8es0_mul
	0000000000001320 T posit8es0_sub
	```
- Reference for implementing 2's complement:
	- [casted with unsigned type, not used](https://stackoverflow.com/questions/25754082/how-to-take-twos-complement-of-a-byte-in-c)

https://www.youtube.com/watch?v=UP-LBRbvI_U