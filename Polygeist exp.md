
https://polygeist.llvm.org/getting_started/Use_Polygeist/

compile with program
https://github.com/llvm/Polygeist/issues/285

note: I cannot get the main output the correct result.

## LLVM Dialect

input:

```cpp
#include <math.h>

double cal_exp(double x)
{
    return exp(x);
}

```

command:

`bin/cgeist -v -S -g -O0 -memref-abi=0 -lm ../test/test.c > test.mlir`

output:

```cpp
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cal_exp(%arg0: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x f64 : (i64) -> !llvm.ptr
    %1 = llvm.mlir.undef : f64
    llvm.store %1, %0 : f64, !llvm.ptr
    llvm.store %arg0, %0 : f64, !llvm.ptr
    %2 = llvm.load %0 : !llvm.ptr -> f64
    %3 = math.exp %2 : f64
    return %3 : f64
  }
}
```


input:

```cpp
#include <math.h>
#include <stdio.h>

double cal_exp(double x)
{
    return exp(x);
}

int main()
{
    printf("%lf", cal_exp(5.32));
}

```

command:

`bin/cgeist -v -S -g -O0 -memref-abi=0 -lm ../test/test.c > test.mlir`

output:

```cpp
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("%lf\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @cal_exp(%arg0: f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.alloca %c1_i64 x f64 : (i64) -> !llvm.ptr
    %1 = llvm.mlir.undef : f64
    llvm.store %1, %0 : f64, !llvm.ptr
    llvm.store %arg0, %0 : f64, !llvm.ptr
    %2 = llvm.load %0 : !llvm.ptr -> f64
    %3 = math.exp %2 : f64
    return %3 : f64
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.320000e+00 : f64
    %0 = llvm.mlir.undef : i32
    %1 = llvm.mlir.addressof @str0 : !llvm.ptr
    %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %3 = call @cal_exp(%cst) : (f64) -> f64
    %4 = llvm.call @printf(%2, %3) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    return %0 : i32
  }
}

```

emit llvm-dialect:

command:

`bin/cgeist -v -S -g -O0 -memref-abi=0 -lm -emit-llvm-dialect ../test/test.c > test.mlir`

```cpp
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("%lf\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @cal_exp(%arg0: f64) -> f64 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x f64 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.undef : f64
    llvm.store %2, %1 : f64, !llvm.ptr
    llvm.store %arg0, %1 : f64, !llvm.ptr
    %3 = llvm.load %1 : !llvm.ptr -> f64
    %4 = llvm.intr.exp(%3)  : (f64) -> f64
    llvm.return %4 : f64
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(5.320000e+00 : f64) : f64
    %1 = llvm.mlir.undef : i32
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %4 = llvm.call @cal_exp(%0) : (f64) -> f64
    %5 = llvm.call @printf(%3, %4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.return %1 : i32
  }
}

```

`bin/mlir-translate -mlir-to-llvmir test.mlir -o test.ll`

```
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str0 = internal constant [4 x i8] c"%lf\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @printf(ptr, ...)

define double @cal_exp(double %0) {
  %2 = alloca double, i64 1, align 8
  store double undef, ptr %2, align 8
  store double %0, ptr %2, align 8
  %3 = load double, ptr %2, align 8
  %4 = call double @llvm.exp.f64(double %3)
  ret double %4
}

define i32 @main() {
  %1 = call double @cal_exp(double 5.320000e+00)
  %2 = call i32 (ptr, ...) @printf(ptr @str0, double %1)
  ret i32 undef
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.exp.f64(double) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

```

`llc --filetype=obj --relocation-model=pic test.ll -o test.o`
`bin/clang -lm test.o -o test.exe`
`./test`

output: `204.383882`


## exp: modify llvm dialect directly

add exp declare, remove custom exp_call

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("%lf\00") {addr_space = 0 : i32}
  llvm.func @exp(%arg0: f64) -> f64
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(5.320000e+00 : f64) : f64
    %1 = llvm.mlir.undef : i32
    %2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
    %4 = llvm.call @exp(%0) : (f64) -> f64
    %5 = llvm.call @printf(%3, %4) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    llvm.return %1 : i32
  }
}
```



`bin/mlir-translate -mlir-to-llvmir test_mod.mlir -o test_mod.ll`
`llc --filetype=obj --relocation-model=pic test_mod.ll -o test_mod.o`
`bin/clang -lm test_mod.o -o test_mod.exe`
`./test_mod.exe`

output:
`204.383882`

if no -lm, clang compiler would 
```bash
/usr/bin/ld: test_mod.o: in function `main':
LLVMDialectModule:(.text+0xa): undefined reference to `exp'
```

if no declare `llvm.func @exp(%arg0: f64) -> f64`
```cpp
test_mod.mlir:9:10: error: 'llvm.call' op 'exp' does not reference a symbol in the current scope
    %4 = llvm.call @exp(%0) : (f64) -> f64
```

# Conclusion

if in mlir, there's 
`llvm.func @exp(%arg0: f64) -> f64` declaration
and then
`%4 = llvm.call @exp(%0) : (f64) -> f64`
with the -lm
you can invoke the exp function in libm