---
marp: true
theme: default
paginate: true
header: 10-22 Personal Research
footer: 
style: "section {

  \  font-size: auto;

  }

  pre, code {\r

  \  background-color: #ffffff;\r   \ 

  \  color: #2d2d2d; \r\ 

  \  padding: 10px;\r   \ 

  \  border-radius: 8px;\r   \ 

  \  font-size: 32px;\r   \ 

  \  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);\ 

  \ }

  h1, h2 {

  \  color: #2a7ae2;

  }"
---

# 10-22 Personal Research  
### by 洪祐鈞
#### 10/22  

---

# Agenda  
1. Introduction  
2. Code Examples  
3. Key Concepts  
4. Summary  

---

# Modifying `KrnlGlobalOp`
- Methodology:
	1. Convert the return type, which is `MemRefType`
	2. Get the value attribute of `KrnlGlobalOp`, which is `DenseElementsAttr`
	3. Modify the `DenseElementsAttr`, Convert the value, which use `attr.mapValues` to get the `APInt`
	4. Write the `APInt` conversion logic into the `mapValues` callback function.
	5. Replace the old operation with new operation with the modified data above.

---

# `Modifying KrnlGlobalOp`

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

# `Modifying KrnlGlobalOp`
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
