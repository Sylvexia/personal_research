洪祐鈞

# Task

- Modify the `KrnlGlobalOp` value attribute.
- Writing a pass that convert all `f32` data type to say, `uint8`
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

# `KrnlGlobalOp`

1. Modify the `memref` return type.
	1. RankedTensorType::get(xShape, builder.getF32Type())
2. From multiple attribute, cast to value attribute, then modify the `denseElementAttr`.
3. For each element, modify the type by type converter and its value.
4. Goal: Mapped to `build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute shape, ::llvm::StringRef name, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr offset, /*optional*/::mlir::IntegerAttr alignment);`

- Input:
```cpp
func.func @test_krnlGlobal(%arg0: f32, %arg1: f32) {
    %1 = "krnl.global"() {name = "constant_2", 
	    shape = [32, 1, 3, 3], 
	    value = dense<"0x2F9C...AB3E"> : 
		    tensor<32x1x3x3xf32>} : () ->
			    memref<32x1x3x3xf32>
  return
}
```
- Command:
	```cpp
	./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=0' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_krnl.mlir
	```
- Output:
```cpp
module {
	func.func @test_krnlGlobal(%arg0: i8, %arg1: i8) {
		%0 = "krnl.global"() {name = "name", 
			shape = [32, 1, 3, 3], 
			value = dense<"0x1423...0315"> : 
				tensor<32x1x3x3xi8>} : () -> 
					memref<32x1x3x3xi8>
		return
	}
}
```

Verification:
code snippet:
```cpp
for (auto [origValue, newValue] : llvm::zip(
		 denseAttr.getValues<APFloat>(), newDenseAttr.getValues<APInt>())) {
  llvm::errs() << "orig: " << origValue.convertToFloat() << "\n";
  // cast to binary
  for (int i = n_bits - 1; i >= 0; i--) {
	llvm::errs() << ((newValue.getZExtValue() >> i) & 1);
  }
  llvm::errs() << "\n";
}
```

log:
```cpp
orig: 3.117385e-01
00010100
orig: 5.425086e-01
00100011
orig: 5.432761e-01
00100011
orig: -1.210481e-01
10001000
orig: 2.717297e-02
00000010
```

Failure:

`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
reason:　arith const failed
```cpp
float value: -INF
error: failed to legalize operation 'arith.constant' that was explicitly marked illegal
another failure:
```
when return
```cpp
func.func @test_krnlGlobalReturn(%arg0: f32, %arg1: f32) -> memref<32x1x3x3xf32> {
    %1 = "krnl.global"() {name = "constant_2", shape = [32, 1, 3, 3], value = dense<"0x2F9C...
    return %1 : memref<32x1x3x3xf32>
}
```