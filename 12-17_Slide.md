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

# 12-17 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Move pass after `scf` pass.
- See how the test pipeline doing?

---

# Can I just put pattern before our pass?

```cpp
  RewritePatternSet prePatterns(&getContext());

  populateAffineToStdConversionPatterns(prePatterns);
  populateSCFToControlFlowConversionPatterns(prePatterns);

  ConversionTarget preTarget(getContext());
  preTarget.addIllegalDialect<affine::AffineDialect, scf::SCFDialect>();
  preTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(module, preTarget, std::move(prePatterns))))
    signalPassFailure();

 // our original pass...
```

---

# Error after remove all custom affine OP pattern

Doing standard with two `applyPartialConversion` seemed to mixed up.

```cpp
/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:16:12: 
error: failed to materialize conversion for 
result #0 of operation 'arith.constant' that remained live after conversion
    %cst = arith.constant 0xFF800000 : f32
           ^
/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:16:12: 
note: see current operation: %1 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32

/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:151:13: 
note: see existing live user here: 
"memref.store"(%1, %364) <{nontemporal = false}> : (f32, memref<f32>) -> ()
            affine.store %cst, %alloca_6[] : memref<f32>
            ^
```

---

# Separate to independent pass.

error: `Only structured control-flow loops are supported.`

- Before `cf`, there's `bufferDeallocation` pass
- Support `std` and `scf` dialect, for preventing memory leak.
	- Match `alloc` with `dealloc`, but consider the control flow.
- In `addKrnlToLLVMPasses`
	- `LowerAffinePass`
	- `BufferDeallocationPass`
	- `ConvertKrnlToLLVMPass`

---

# Solution

Create a separate pass.

```cpp
populateAffineToStdConversionPatterns(patterns);
target.addIllegalDialect<mlir::affine::AffineDialect>();
target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
```

- Meaning: We do not need to lower control flow related operation 1 by 1.

---

# Testing Pipeline

- Basically we just need 2 script:
	- RunONNXModel.py
	- RunONNXModelZoo.py

---

# RunONNXModel.py

- load `onnx` model.
- compile with `onnx-mlir`
- using python runtime to load the share object and run with `OMExecutionSession`
- verify the output

---

# RunONNXModelZoo.py

- clone `Github` and download dataset and model.
	- dataset is .pb file, which is further used as numpy input
- populate RunONNXModel.py options
	- options = `compile_args` + dataset + model
- Log out the result

---

# preserveLLVMIR

```cpp
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:
64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@constant_8_mnist_posit = 
internal constant [1280 x i8] c"*#\9A*)\A3(\A6\9E\9B\B1\1B\10...

declare i32 @strncmp(ptr, ptr, i64)

declare void @omGetExternalConstantAddr(ptr, ptr, i64)

; Function Attrs: memory(none)
declare i1 @posit8es2_ogt(i8, i8) #0
```
---

# preserveLLVMIR

```cpp
define { ptr, ptr, i64, [2 x i64], [2 x i64] } 
@main_graph_mnist_posit(ptr %0, ptr %1, i64 %2, i64 %3,
 i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = call ptr @malloc(i64 add (i64 ptrtoint 
  (ptr getelementptr (i8, ptr null, i32 25088) to i64), i64 16))

93:                                               ; preds = %40
  %94 = getelementptr i8, ptr @constant_1_mnist_posit, i64 %31
  %95 = load i8, ptr %94, align 1
  %96 = call i8 @posit8es2_add(i8 %42, i8 %95)

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_mnist_posit(ptr %0)
  ret ptr %2
}
```

---

# Conclusion

- Thinking of a scheme to do testcase in python
	- Move posit conversion into Python
	- Import dataset to train is mostly done by python, we need to code in python?
- Bypass `emit_c` attribute injection.