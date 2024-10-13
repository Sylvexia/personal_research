# Task

- `MLIR`
	- Modify the `KrnlGlobalOp` value attribute.
	- Dispatch the type.
		- `./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=16 es-val=2' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir`
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
	- Turn off the constant propagation in posit.
	- `@run_main_graph`
		- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.
# Quantize inspiration

- `Tensorflow` inspiration:
	- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/tests/quantize.mlir
		- quantize bias:
			```cpp
			// ----
			// A requantized conv2D with fused bias.
			// CHECK-LABEL: @conv2d_bias_requantize
			// CHECK: %cst = constant dense<tensor<3x5xi8>
			// CHECK-NEXT: %cst_0 = constant dense<tensor<5xi32>
			// CHECK-NEXT: %0 = "quant.qcast"(%arg0) : (tensor<300x3xf32>) -> tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>
			// CHECK-NEXT: %1 = "quant.scast"(%cst) : (tensor<3x5xi8>) -> tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>
			// CHECK-NEXT: %2 = "quant.scast"(%cst_0) : (tensor<5xi32>) -> tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>
			// CHECK-NEXT: %3 = "intelquant.real_conv2d_bias_requantize"(%0, %1, %2) : (tensor<300x3x!quant.uniform<u8:f32, 0.037564418067230126:163>>, tensor<3x5x!quant.uniform<u8:f32, 0.0062823070315864236:127>>, tensor<5x!quant.uniform<i32:f32, 0.0629921259842528>>) -> tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>
			// CHECK-NEXT: %4 = "quant.dcast"(%3) : (tensor<300x5x!quant.uniform<i8:f32, 0.0629921259842528:-1>>) -> tensor<300x5xf32>
			func @conv2d_bias_requantize(%arg0: tensor<300x3xf32>) -> tensor<300x5xf32> {
			  %0 = "quant.stats"(%arg0) {layerStats: dense<tensor<2xf32>, [-6.123e+00, 3.45e+00]>} : (tensor<300x3xf32>) -> tensor<300x3xf32>
			  %cst = constant  {name: "constant.35"} dense<tensor<3x5xf32>, [[-1.060230e-01, 1.215050e-01, 8.002390e-01, -7.688850e-01, 0.0966112986], [6.890140e-01, -4.070560e-01, -0.797852993, 3.789250e-03, -2.088810e-01], [-6.085290e-01, 2.766170e-02, 2.685570e-01, 5.774010e-01, -4.284370e-01]]>
			  %cst_0 = constant  {name: "constant.37"} dense<tensor<5xf32>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]>
			  %1 = "intelquant.real_conv2d_bias_requantize"(%0, %cst, %cst_0) : (tensor<300x3xf32>, tensor<3x5xf32>, tensor<5xf32>) -> tensor<300x5xf32>
			  %2 = "quant.stats"(%1) {layerStats: dense<tensor<2xf32>, [-8.000000e+00, 8.000000e+00]>} : (tensor<300x5xf32>) -> tensor<300x5xf32>
			  return %2 : tensor<300x5xf32>
			}
			```
	- Note: Currently `tensorflow` seems like it does not do the quantization using MLIR. (?)
	- [RFC](https://discourse.llvm.org/t/rfc-improvements-in-the-quant-dialect/79942)
		- Lowering `qcast`, `scast`, `dcast`
		- https://github.com/llvm/llvm-project/pull/100667
		- example: (Pass `--lower-quant-ops`)
			```cpp
			!qalias = !quant.uniform<i8<-8:7>:f32, 2.0:10>
			func.func @f(%arg0: tensor<3x5xf32>) -> tensor<3x5x!qalias> {
				%0 = quant.qcast %arg0 : tensor<3x5xf32> to tensor<3x5x!qalias>
				return %0 : tensor<3x5x!qalias>
			}
			```
			- types: `!qalias`
			- map to: i8
				- range: -8 ~ 7
			- original: f32
			- scaling factor: 2.0, zero point: 10
		- output:
			```cpp
			func.func @f(%arg0: tensor<3x5xf32>) -> tensor<3x5x!qalias> {
			    // Create scale tensor.
				// NOTE: All 'arith.constant' + 'tensor.splat' ops will be canonicalized into
				// a single 'arith.constant' for statically shaped tensors.
				%cst = arith.constant 2.000000e+00 : f32
				%splat = tensor.splat %cst : tensor<3x5xf32>
				
				// Divide by scale
				%0 = arith.divf %arg0, %splat : tensor<3x5xf32>
				
				// Create zero point float tensor
				%c10_i8 = arith.constant 10 : i8
				%splat_0 = tensor.splat %c10_i8 : tensor<3x5xi8>
				%1 = arith.sitofp %splat_0 : tensor<3x5xi8> to tensor<3x5xf32>
				
				// Add zero point
				%2 = arith.addf %0, %1 : tensor<3x5xf32>
				
				// Convert stored value to integer
				%3 = arith.fptosi %2 : tensor<3x5xf32> to tensor<3x5xi8>
				
				// Clamp stored value
				%c-8_i8 = arith.constant -8 : i8
				%c7_i8 = arith.constant 7 : i8
				%splat_1 = tensor.splat %c-8_i8 : tensor<3x5xi8>
				%splat_2 = tensor.splat %c7_i8 : tensor<3x5xi8>
				%4 = arith.maxsi %3, %splat_1 : tensor<3x5xi8>
				%5 = arith.minsi %4, %splat_2 : tensor<3x5xi8>
				
				// Cast stored value to quantized type
				%6 = quant.scast %5 : tensor<3x5xi8> to tensor<3x5x!qalias>
				return %6 : tensor<3x5x!qalias>
				}
			```
- Keywords
	- `llvm`
		- `quantizeFloatToInt`
		- `quant-convert-const`
	- `tpu-mlir`
		```cpp
		quant::UniformQuantizedType getUniformQuantizedType(Value v) {
		  return v.getType()
		      .cast<RankedTensorType>()
		      .getElementType()
		      .cast<quant::UniformQuantizedType>();
		}
		```

# POSIT Value Conversion

- API spec: 
	- `uint64_t convertFloat32ToPosit(uint64_t raw_bit, uint8_t n_bits, uint8_t es_val) {`
- 

# MLIR Type Dispatcher


# MLIR structure

The following MLIR is before lower to `llvm dialect`
- `--EmitMLIR - Lower the input to MLIR built-in transformation dialect.`
	- command:
		- `./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
	- output:
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
	- Attributes:
		- value
			- Not stored in external files:
				- DenseResourceElementsAttr
				- DenseElementsAttr
			- Stored in external files
				- If were
		- offset
			- memory offset from the base address of the global buffer
			- `memref.reinterprete_cast`:
				- https://discourse.llvm.org/t/question-about-memref-reinterpret-casts-offset/76082
		- alignment
			- https://hackmd.io/@sysprog/c-memory#data-alignment
			- `memref.global`
				- [nobody needs it](https://discourse.llvm.org/t/alignment-on-memref-global/3381)
					- so ONNX-MLIR creates a custom one?
	- Probably we don't need to touch offset and alignment.
- `KrnlGlobalOp` creation
	- created by `KrnlBuilder::constant`
	- code snippet:
		```cpp
		Value KrnlBuilder::constant(MemRefType type, StringRef name,
		    std::optional<Attribute> value, std::optional<IntegerAttr> offset,
		    std::optional<IntegerAttr> alignment) const {
		  static int32_t constantID = 0;
		  return b().create<KrnlGlobalOp>(loc(), type,
		      b().getI64ArrayAttr(type.getShape()),
		      b().getStringAttr(name + std::to_string(constantID++)),
		      value.value_or(nullptr), offset.value_or(nullptr),
		      alignment.value_or(nullptr));
		}
		```
	- 

# Posit Wrapper verification and 2's complement issue

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