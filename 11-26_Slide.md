---
marp: true
theme: default
paginate: true
header: 
footer: 
style: "h1, h2, h3 {\r  text-align: center;\r}

  pre, code {\r  background-color: #ffffff;\r    \r  color: #2d2d2d; \r  \r  font-size: auto;\r }\r

  section {\r  font-size: auto;\r}\r

  img[alt~=\"center\"]\ 

  {\r  display: block;\r  margin: 0 auto;\r}"
---

# 11-26 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- In posit wrapper, using token pasting for `n_bits, es_val` operation.
- In `onnx-mlir`
	- Add our custom pass to main compiler, with `n_bits, es_val` config
	- Successfully lowered to `.so`, but there's issue.
		- Fixed the `krnl.globalOp` name duplication. (dedicated for model weight)
		- New issue:
			- Generated function declaration is altered later in `FuncToLLVM` pass
		- Function declaration generation needs to be revised.

---

# Macro-ed wrapper (Proof of Concept)

Once we have generalized result, we can macro it

```cpp
uint8_t posit8es0_add(uint8_t a, uint8_t b) {
  auto pa = get_posit<8, 0>(a);
  auto pb = get_posit<8, 0>(b);
  auto pc = pa + pb;
  uint8_t res = get_uType<8, 0, uint8_t>(pc);
  return res;
}
```

---

# Macro-ed wrapper (Proof of Concept)

- using token pasting `##` in c

```cpp
#define SOURCE_POSIT_ADD_FUNC(bits, es_val)                                    \
  uint##bits##_t posit##bits##es##es_val##_add(uint##bits##_t a,               \
                                               uint##bits##_t b) {             \
    auto pa = get_posit<bits, es_val>(a);                                      \
    auto pb = get_posit<bits, es_val>(b);                                      \
    auto pc = pa + pb;                                                         \
    uint##bits##_t res = get_uType<bits, es_val, uint##bits##_t>(pc);          \
    return res;                                                                \
  }

SOURCE_POSIT_ADD_FUNC(8, 0)
SOURCE_POSIT_ADD_FUNC(16, 1)
```

---

# Macro-ed wrapper (Proof of Concept)

- verified with `nm` that has simple symbol name:

```
nm c_api/custom/posit/libposit_c_api_custom.a | grep 16
00000000000021c0 T posit16es1_add
```

---

# Adding our Pass

- Pass Order:
	- `ONNXToMLIR` $\rightarrow$ `ONNXToKrnl` $\rightarrow$ `KrnlToAffine` $\rightarrow$ `KrnlToLLVM`
		1. Represent ONNX in MLIR
		2. Decompose ONNX with custom representation.
		3. Custom loop representation to affine.
		4. Populate standard conversion with additional lowering `krnl` operator at the end.
- The pass is added immediately after the `KrnlToAffine`

---

# Adding our Pass

- Two main compiler:
	- `onnx-mlir-opt`: For testing pass separately.
	- `onnx-mlir`: Main compiler, putting all pass together that we can end to end compile.
- Basically for past month, we run our pass `onnx-mlir-opt` and test separately
- Now we can (kind of) execute, compile from `onnx` to `mlir`
	- `./onnx-mlir --EmitMLIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx`

---

# Compilation process in the main compiler

- `compileModuleToSharedLibrary`: `.mlir` -> `.so`
    - `compileModuleToObject`: `.mlir` -> `.o`
    	- `genLLVMBitcode`: MLIR -> LLVM bitcode
    		- `translateModuleToLLVMIR`: translate MLIR to LLVMIR
    		- `tailorLLVMIR`: add things that not in MLIR but LLVM
    		- using `opt` to optimize bitcode
    	- `genModelObject`: using `llc` to compile LLVM bitcode to object file.
    - `genSharedLib`: using `cxx` to compile and link e.g. `cruntime`, `jniruntime`
    	- (Main compiler has option to config custom -l and -L)

---

# Test Failed (ciface issue)

- This is message from building test.
```cpp
/usr/bin/ld: /home/sylvex/onnx-mlir/build/docs/doc_example/libadd.so: 
undefined reference to `_mlir_ciface_posit8es8_add'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```
- It make me look into the compiled shared library and MLIR file

---

# Test Failed (ciface issue)

- Dumping shared library: `nm LIB.so`
	```cpp
	00000000000039b0 T _mlir_ciface_main_graph_lib
                 U _mlir_ciface_posit8es8_add
                 U _mlir_ciface_posit8es8_mul
                 U _mlir_ciface_posit8es8_oge
                 U _mlir_ciface_posit8es8_ogt
                 U _mlir_ciface_posit8es8_select
	```

---

# Test Failed (ciface issue)

- MLIR:
```cpp
llvm.func private @posit8es8_mul(%arg0: i8, %arg1: i8) 
  -> i8 attributes {llvm.emit_c_interface, 
  memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, 
  sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_posit8es8_mul(%arg0, %arg1) : (i8, i8) -> i8
    llvm.return %0 : i8
  }
  llvm.func @_mlir_ciface_posit8es8_mul(i8, i8) 
    -> i8 attributes {llvm.emit_c_interface, sym_visibility = "private"}

```
- It raises a question: what is ciface?

---

## What is ciface?

- `ONNXToMLIR` -> `ONNXToKrnl` -> `KrnlToAffine` -> `KrnlToLLVM`
	- Where does the `_ciface_` get added?
	- Not in `KrnlToAffine`
	- In `KrnlToLLVM`, I suspect that the pass "`FuncToLLVM`" add `_mlir_ciface_` prefix to function declaration symbol.
- Clue:
	- See runtime function `@omTensorGetDataPtr` generation, it does not have `_mlir_ciface_`
- Future path:
	- Revise our function generating scheme.
	- Move our pass after `_mlir_ciface_` generation.

---

# Conclusion:

- Revise our function generation scheme.
- Look deeper into runtime.