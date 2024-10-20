---
marp: true
theme: default
paginate: true
header: 
footer: 
style: "section {

  \  font-size: auto;

  }

  pre, code {\r

  \  background-color: #ffffff;\r   \ 

  \  color: #2d2d2d; \r\ 

  \  padding: 10px;\r   \ 

  \  border-radius: 8px;\r   \ 

  \  font-size: 32px;\r

  \ }

  h1, h2 {

  \  color: #2a7ae2;

  }"
---

# 10-22 Personal Research  
### by 洪祐鈞
#### 10/22  

---

# Summary
1. Successfully modify the `KrnlGlobalOp` type and value.

---

# Modifying `KrnlGlobalOp`
- Methodology:
	1. Convert the return type, which is `MemRefType`
	2. Get the value attribute of `KrnlGlobalOp`, which is `DenseElementsAttr`
	3. Modify the `DenseElementsAttr`, Convert the value, which use `attr.mapValues` to get the `APInt`
	4. Write the `APInt` conversion logic into the `mapValues` callback function.
	5. Replace the old operation with new operation with the modified data above.

---

# Modifying`KrnlGlobalOp`

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
	```bash
	./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=0' ./test_krnl.mlir
	```

---

# Modifying `KrnlGlobalOp`
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

---

# Modifying `KrnlGlobalOp`

- Verification output: 
	- This is posit<32, 2>, different from the aforementioned posit<8,0>
		```
		original float value: -3.941769e-01
		original float raw bit:  1 01111101 10010011101000110001111
		new raw bit: 10110100100111010001100011110000
		```
	- verify with posit tool
		`./posit -3.941769e-01 10110100100111010001100011100000`

- Compare:
	```cpp
	10110100100111010001100011110000 // mlir log
	10110100100111010001100011100000 // posit tool
	```
	- One bit is off?
---

# Modifying `KrnlGlobalOp`

- compare:
	```cpp
	10110100100111010001100011110000 // mlir log
	10110100100111010001100011100000 // posit tool
	```
	- Notice that we feed the `-3.941769e-01` directly into posit tool! 
		- This number already lost the precision
	- This is why we also need to log original float raw bit:  
		- `1 01111101 10010011101000110001111`
---
# Modifying `KrnlGlobalOp`

- Bug and Resolve:
	- The `addConversion` input type might actually matters
		```cpp
		addConversion([bitWidth](FloatType type) -> Type {
		  if (isa<Float32Type>(type)) {
		    return IntegerType::get(
		      type.getContext(), bitWidth, IntegerType::Signless);
		}
		```
	- ``([bitWidth](Type type)`` would make the `FloatType` information lost.
	- If you want to `([bitWidth](Type type)`
		- You need to `dyn_cast` the type

---
# Modifying `KrnlGlobalOp`

- Bug and Resolve:
	- The order of the `addConversion` also matters:
		1. `addConversion([](Type type)`
		2. `addConversion([bitWidth](MemRefType type)`
		3. `addConversion([bitWidth](TensorType type)`
		4. `addConversion([bitWidth](FloatType type)`

	- The first one accept any type and return the original
	- This act as a fallback mechanics for not able to convert all by once.

---
# Modifying `KrnlGlobalOp`

For pattern `addConversion([](Type type)` return the same type as fallback mechanics
- You can see the same pattern in `onnx-mlir` project.
- And its comment:
	`The order of type conversion is important: later ones are tried earlier.`
	- No relevant information are found in the documentation.

---
# Modifying `KrnlGlobalOp`

- Revise:
	```cpp
	bool res = typeConverter.isSignatureLegal(op.getFunctionType()) &&
			   typeConverter.isLegal(&op.getBody());
	```
- `getBody` means all the operation inside the body must be integer types

---
# Modifying `KrnlGlobalOp`
