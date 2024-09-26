
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