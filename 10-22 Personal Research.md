洪祐鈞

# `Modifying KrnlGlobalOp`

- Methodology:
	1. Convert the return type, which is `MemRefType`
	2. Get the `value` attribute of `KrnlGlobalOp`, which is `DenseElementsAttr`
	3. Modify the `DenseElementsAttr`:
		1. Convert the value, which use `attr.mapValues` to get the `APInt`
			1. Can refer to `UniformQuantizedPerAxisValueConverter::convert` in `LLVM quant` dialect
	4. Write the `APInt` conversion logic into the `mapValues` callback function.
	5. Replace the old operation with new operation with the new `MemRefType` and `DenseElementsAttr`

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

```cpp
func.func @test_krnlGlobalReturn(%arg0: f32, %arg1: f32) 
  -> memref<32x1x3x3xf32> 
  {
    %1 = "krnl.global"() 
    {
      name = "constant_2", 
      shape = [32, 1, 3, 3], 
      value = dense<"0x2F9C...AB3E"> : tensor<32x1x3x3xf32>
    } : () -> memref<32x1x3x3xf32>
  return %1 : memref<32x1x3x3xf32>
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

```cpp
func.func @test_krnlGlobalReturn(%arg0: i8, %arg1: i8) 
  -> memref<32x1x3x3xi8> 
  {                                                                        %0 = "krnl.global"() 
    {
      name = "name", 
      shape = [32, 1, 3, 3], 
      value = dense<"0x1423...0315"> : tensor<32x1x3x3xi8>
    } : () -> memref<32x1x3x3xi8>                                          return %0 : memref<32x1x3x3xi8>                                      }
}
```
- Observation:
	- In the testcase, the following is modified properly
		- `f32` -> `i8` 
			- input argument
		- `memref<32x1x3x3xf32>` -> `memref<32x1x3x3xi8>` 
			- function return type and `KrnlGlobalOp` return type
		- `value = dense<"0x2F9C...AB3E"> : tensor<32x1x3x3xf32>` -> `dense<"0x1423...0315"> : tensor<32x1x3x3xi8>`
			- `KrnlGlobalOp` `value` attribute value and type
	- What's inside those cryptic bits?
		- Flattened the array, and concatenate the element with hexadecimal representation.
			- Should not have precision loss.
		- Dense attribute number of character: (without `"dense<0x" ">"`)
			- old: 2304 (`f32`)
			- new: 576 (`i8`), which is 100% -> 25%
- Verification:
	- From old and new `denseAttr`, iterate at the same time and compare them. 
```cpp
for (auto [origValue, newValue] : llvm::zip(
  denseAttr.getValues<APFloat>(), 
  newDenseAttr.getValues<APInt>())) {
  
  llvm::errs() << "original float value: " 
    << origValue.convertToFloat() << "\n";

  llvm::errs() << "original float raw bit: ";
  uint64_t orig_raw_bit = origValue.bitcastToAPInt().getZExtValue();
  for(int i = 31; i >= 0; i--) {
	if (i == 30 || i == 22) {
	  llvm::errs() << " ";
	}
	llvm::errs() << ((orig_raw_bit >> i) & 1);
  }
  llvm::errs() << "\n";

  llvm::errs() << "new raw bit: ";
  uint64_t raw_bit = newValue.getZExtValue();
  for (int i = n_bits - 1; i >= 0; i--) {
	llvm::errs() << ((raw_bit >> i) & 1);
  }
  llvm::errs() << "\n";
}
```

verification output:
```
original float value: -3.941769e-01
original float raw bit:  1 01111101 10010011101000110001111
new raw bit: 10110100100111010001100011110000
```

- verify with posit tool
`./posit -3.941769e-01 10110100100111010001100011100000`

- compare:
```cpp
10110100100111010001100011110000 // mlir log
10110100100111010001100011100000 // posit tool
```

- Bug and Resolve:
	- The `addConversion` input type might actually matters
```cpp
addConversion([bitWidth](FloatType type) -> Type {
  if (isa<Float32Type>(type)) {
	return IntegerType::get(
		type.getContext(), bitWidth, IntegerType::Signless);
  }
```

``([bitWidth](Type type)`` would make the `FloatType` information lost.
If you want to `([bitWidth](Type type)`
You need to `dyn_cast` the type

The order of the `addConversion` also matters:

`addConversion([](Type type)`
`addConversion([bitWidth](MemRefType type)`
`addConversion([bitWidth](TensorType type)`
`addConversion([bitWidth](FloatType type)`

The first one accept any type and return the original
This act as a fallback mechanics for not able to convert all by once.
You can see the same pattern in `onnx-mlir` project.
And its comment: `The order of type conversion is important: later ones are tried earlier.`
- No relevant information are found in the documentation.

Revise:
```cpp
bool res = typeConverter.isSignatureLegal(op.getFunctionType()) &&
		   typeConverter.isLegal(&op.getBody());
```
`getBody` means all the operation inside the body must be integer types

Failure:
- Adding pass to the main compiler currently does not work
	- Might be not dealing with other `memref` operations.

`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
reason:　arith const failed
```cpp
float value: -INF
error: failed to legalize operation 'arith.constant' that was explicitly marked illegal
another failure:
```