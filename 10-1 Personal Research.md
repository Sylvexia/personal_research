Task:
1. func to llvm dialect to llvm ir and make it executable with libm
	1. onnx-mlir-opt --convert-xxx-pass
		1. Need to see the real mlir conversion
3. What is extern C? or how to create a c wrapper around C++?
	1. See universal library
4. How to get the symbol string of a library?
	1. c should be the same
	2. nm
5. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

# Summary

- For `llvm` dialect, take invoking `exp` for example, 
	- we can now:
		- lower from math dialect 
		- to `func` dialect 
		- to `llvm` dialect 
		- to `llvm` `IR`
		- to executable and execute it.
	- For `math` to `func` creation
		- We need to know how to create `func` at `llvm` dialect. 
			- Then backtrack to `func` dialect.
		- We verify this in `Polygeist`, which convert C to `llvm` dialect.
		- `llvm` `func` creation:
			- If we can create the declaration, invoke and proper linking:
				- Declaration: `llvm.func @exp(%arg0: f64) -> f64`
				- Invoke: `%4 = llvm.call @exp(%0) : (f64) -> f64`
				- Linking: Link the math for example.
			- With: `mlir-translate`, `llc` and `clang`, we can convert to executable:
				- `bin/mlir-translate -mlir-to-llvmir test_mod.mlir -o test_mod.ll`
				- `llc --filetype=obj --relocation-model=pic test_mod.ll -o test_mod.o`
				- `bin/clang -lm test_mod.o -o test_mod.exe`
			- We can `./test_mode.exe` and get the output, the experiment is as below.
- For symbol name
	- In C it's the same as function call.
	- In C++ there's will have name mangling inevitably, since it must support function overloading, namespace, class...
		- Unless you specify `extern "C"`
- In Universal number library, it's mainly C++.
	- There's C wrapper, but it's mostly macro generated c function name, and hard to get the es value settings. (nbits and es is preset like (8, 0), (16, 1), (32, 2))
	- We probably need to implement ourselves, I failed to see a way to modify the macro.
	- Rough prototype:
		- 
- For converting the 
- We need to convert

# Polygeist experiment to get lower c to link libm

[[Polygeist exp]]

For c lowering to mlir to llvm to executable.

# From Math dialect to Func Dialect to LLVM dialect to LLVMIR to executable.

Command

`func_to_llvm.mlir`:

```cpp
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("%lf\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(6.9e+00 : f64) : f64
    %1 = llvm.mlir.undef : i32
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %4 = math.exp %0 : f64
    %5 = llvm.call @printf(%3, %4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.return %1 : i32
  }
}
```

`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt /home/sylvex/onnx-mlir/src/math_to_llvm.mlir --convert-custom-math-to-func -o func_to_llvm.mlir -allow-unregistered-dialect`
	Allow the unregistered dialect.

`/home/sylvex/Polygeist/build/bin/mlir-opt --convert-func-to-llvm func_to_llvm.mlir -o lowered.mlir`

`/home/sylvex/Polygeist/build/bin/mlir-translate -mlir-to-llvmir lowered.mlir -o lowered.ll`

`/home/sylvex/Polygeist/build/bin/llc --filetype=obj --relocation-model=pic lowered.ll -o lowered.o`

`/home/sylvex/Polygeist/build/bin/clang -lm lowered.o -o lowered.exe`

`./lowered.exe`

output: 992.274716

# MLIR func mapping

Declaration:

```cpp
auto opFunctionTy = FunctionType::get(
	rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
opFunc = rewriter.create<func::FuncOp>(
	rewriter.getUnknownLoc(), name, opFunctionTy);
```

Oh no, errors!

`./onnx-mlir-opt /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir --convert-arith-to-posit-func`

```bash
/home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir:2:8: error: failed to materialize conversion for result #0 of operation 'arith.addf' that remained live after conversion
  %0 = arith.addf %arg0, %arg1 : f32
       ^
/home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir:2:8: note: see current operation: %1 = "arith.addf"(%arg0, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
/home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test.mlir:3:3: note: see existing live user here: func.return %1 : f32
  return %0 : f32
  ^
```

```cpp
// File: ConvertF32ToF16Pass.cpp

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct ConvertF32ToF16Pass : public PassWrapper<ConvertF32ToF16Pass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](Operation *op) {
      // Convert types
      for (auto &result : op->getResults()) {
        if (result.getType().isF32()) {
          result.setType(builder.getF16Type());
        }
      }

      // Convert operands
      for (auto &operand : op->getOpOperands()) {
        if (operand.get().getType().isF32()) {
          operand.set(builder.create<arith::ExtFOp>(op->getLoc(), builder.getF16Type(), operand.get()));
        }
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createConvertF32ToF16Pass() {
  return std::make_unique<ConvertF32ToF16Pass>();
}

static PassRegistration<ConvertF32ToF16Pass> pass("convert-f32-to-f16", "Convert all f32 types and values to f16");
```