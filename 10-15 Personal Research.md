洪祐鈞
# Task

- `MLIR`
	- Modify the `KrnlGlobalOp` value attribute.
	- Dispatch the type.
		- `./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=16 es-val=2' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir`
		- struct `FrontendToKrnlLoweringPass` can be referred
	- Writing a pass that convert all `f32` data type to say, `uint8`
	- Get to know the function `private` and `readonly` attribute.
		- [ref](https://llvm.org/docs/LangRef.html)
		- `private`
			- Global values with “`private`” linkage are only directly accessible by objects in the current module. In particular, linking code into a module with a private global value may cause the private to be renamed as necessary to avoid collisions. Because the symbol is private to the module, all references can be updated. This doesn’t show up in any symbol table in the object file.
		- `readonly`
			- This attribute indicates that the function does not write through this pointer argument, even though it may write to the memory that the pointer points to.
			- If a function writes to a `readonly` pointer argument, the behavior is undefined.
	- what does `MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` mean?
	- Turn off the constant propagation in posit.
	- `@run_main_graph`
		- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.

# POSIT Value Conversion in MLIR

- API spec: 
	- `uint64_t convertFloat32ToPosit(uint64_t raw_bit, uint8_t n_bits, uint8_t es_val)`
		- `raw_bit` input is 64-bit correspond with `apFloat.bitcastToAPInt().getZExtValue()` export datatype, however the `hsb` 32-bit is all zero.
		- output is also 64-bit is because to bridge the `mlir` `rewriter.getIntegerAttr(IntType, uintValue)`
	- Bug Resolved:
		- Notice that the raw data from `apfloat` is interpreted with f32 instead of f64, takes half a day to debug.

# MLIR Posit Config Type Dispatcher

- Summary:
	- We can config the `nbits` and `es_val` with command line!
		- Command:
			- pass the argument `n-bits` and `es-val` to the pass argument.
			- `./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=16 es-val=2' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir`
	- It would
		- Generate the function call symbol based on `n-bits`, `es-val`
		- Dispatch the type based on `n-bits`
- Methodology:
	- Accept the command line argument:
		- Add the following to your pass initialization:
		- code snippet
			```cpp
			  Option<int> _n_bits{*this, "n-bits",
		      llvm::cl::desc("Number of bits in posit"), llvm::cl::init(8)};
		  Option<int> _es_val{*this, "es-val",
		      llvm::cl::desc("Number of bits in exponent"), llvm::cl::init(0)};
			```
	- Dispatch the type:
		- Utilize `TypeConverter` and lambda function.
			- If we substitute the f32 to posit, just substitute the f32 part and all will apply
		- code snippet:
			```cpp
			struct FloatToIntTypeConverter : public mlir::TypeConverter {
			  explicit FloatToIntTypeConverter(uint8_t bitWidth) {
			    addConversion([bitWidth](Type type) -> Type {
			      if (isa<Float32Type>(type)) {
			        return IntegerType::get(
			            type.getContext(), bitWidth, IntegerType::Signless);
			      }
			      return type;
			    });
			...
			```
- Example output:
```cpp
// --convert-arith-to-posit-func='n-bits=16 es-val=2'
func.func @test_arith_const(%arg0: i16, %arg1: i16) {                         
  %c16384_i16 = arith.constant 16384 : i16                                    
  %c18432_i16 = arith.constant 18432 : i16                                    
  %0 = call @posit16es2_add(%arg0, %c18432_i16) : (i16, i16) -> i16           
  return                                                                    
} 
```

```cpp
// --convert-arith-to-posit-func='n-bits=32 es-val=1'
func.func @test_arith_const(%arg0: i32, %arg1: i32) {
  %c1073741824_i32 = arith.constant 1073741824 : i32
  %c1342177280_i32 = arith.constant 1342177280 : i32
  %0 = call @posit32es1_add(%arg0, %c1342177280_i32) : (i32, i32) -> i32
  return
} 
```

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
	- `alloc` allocate memory on heap.
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
	- TLDR:
		- Convert value attribute `denseElementAttr` should just be fine.
	- Attributes:
		- value
			- Not stored in external files:
				- `DenseResourceElementsAttr`
					- For large binary object (blob)
					- Only created when NNPA accelerator.
					- Example: 
						```cpp
							func.func @constant_dense_resource() { dense<[0.203224242, -0.254296064, -0.365104556, -0.469196141, 0.466041982]> : tensor<5xf32> : !spirv.array<5 x f32>
							  %0 = arith.constant dense_resource<dense_resource_test_5xf32> : tensor<5xf32>  
							  %1 = arith.constant dense_resource<dense_resource_test_2xi32> : vector<2xi32>  
							  %2 = arith.constant dense_resource<dense_resource_test_2x2xf32> : tensor<1x2x2xf32>  
							  return
							  }
							}
							
							{-#
							  dialect_resources: {
							    builtin: {
							      dense_resource_test_2xi32: "0x400000000100000002000000",
							      dense_resource_test_5xf32: "0x08000000041A503E183382BEFCEEBABE7A3AF0BE0E9DEE3E",
							      dense_resource_test_2x2xf32: "0x0800000054A3B53ED6C0B33E55D1A2BDE5D2BB3E"
							    }
							  }
							#-}
						```
				- `DenseElementsAttr`
					- Our main goal, more of the information can be found below.
			- Stored in external files
				- `store-constants-to-file`
					- Constants will be stored on a binary file instead of be embedded into the `model.so` when compiling a big model.
		- offset
			- memory offset from the base address of the global buffer?
			- `memref.reinterprete_cast`:
				- https://discourse.llvm.org/t/question-about-memref-reinterpret-casts-offset/76082
		- alignment
			- https://hackmd.io/@sysprog/c-memory#data-alignment
			- `memref.global`
				- [nobody needs it](https://discourse.llvm.org/t/alignment-on-memref-global/3381)
					- so ONNX-MLIR creates a custom one?
	- Probably we don't need to touch offset and alignment.
- `KrnlGlobalOp` creation
	- Only created by `KrnlBuilder::constant`
		- code snippet:
			```cpp
			Value KrnlBuilder::constant(MemRefType type, StringRef name,
			    std::optional<Attribute> value, std::optional<IntegerAttr> offset,
			    std::optional<IntegerAttr> alignment) const {
			  static int32_t constantID = 0;
			  return b().create<KrnlGlobalOp>(loc(), type,
			      b().getI64ArrayAttr(type.getShape()),
			      b().getStringAttr(name + std::to_string(constantID++)),
			      value.value_or(nullptr), offset.value_or(nullptr),S
			      alignment.value_or(nullptr));
			}
			```
	- get `DenseElementAttribute` from value:
		- code snippet:
			```cpp
			DenseElementsAttr getDenseElementAttributeFromKrnlValue(Value value) {
			  KrnlGlobalOp globalOp =
			      dyn_cast_or_null<mlir::KrnlGlobalOp>(value.getDefiningOp());
			  if (globalOp)
			    if (globalOp.getValue().has_value())
			      return mlir::dyn_cast<DenseElementsAttr>
				      (globalOp.getValueAttr());
			
			  return nullptr;
			}
			```
- `DenseElementsAttr` creation:
	- code snippet:
		```cpp
		DenseElementsAttr cats_int64s = mlir::DenseElementsAttr::get(
        RankedTensorType::get(
            cats_int64sAttr.size(), rewriter.getIntegerType(64)),
        cats_int64sAttr.getValue());
		```
# Universal Posit Wrapper verification and 2's complement issue

- In universal library, the negative is 2's complement except the sign bit compare to standard
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
- Verify the library name (since we still have `api` written in `c++`)
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