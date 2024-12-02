
# Summary

- Unified way to add operation in posit wrapper
	- Supported all posit config of $+$, $-$, $\times$, $\div$, select, compare
	- Verified how to link the library with custom c example.
- Trying to work with end-to-end flow
	- Linked the wrapper to the MNIST model
	- Compiled with user driver code.
	- We can execute and get the model result.
	- Result is currently not verified.

# How to compile model to library

1. Set Environment variable
2. Using `onnx-mlir`, to compile`model.onnx` -> `model.so`
3. Using `g++`, to compile `userDriver.cpp` + `model.so` -> `run.exe`
4. You can execute and get the result from `run.exe`

command:

1. `export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH`
2. `onnx-mlir -EmitLib mnist.onnx`
3. `g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE`
4. `./mnist`

---

Assume env variable is properly set up

```
export ONNX_MLIR_ROOT=$(pwd)/../..
export ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin
export ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include
export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH
export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
```

1. `onnx-mlir -EmitLib mnist.onnx`
2. `g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE`
3. `./mnist`

---
# What should do to include posit?

- 2 compiler, `onnx-mlir` and `g++`, for linking
- Intuitively, we should look into what `onnx-mlir` do
	1. MLIR -> LLVM
	2. LLVM -> `.bc`
	3. `.bc` -> `.o` using `llc`
	4. `.o` -> `.so` using `clang++`
- We can log and get the `llc` and `clang++` command with -v in `onnx-mlir` (only important flag is shown here)
	- `llc -filetype=obj -o model.o model.bc`
	- `clang++ model.o -o model.so -shared -L/home/sylvex/onnx-mlir/build/Debug/lib -lcruntime`
- Clue: 
	- Using -l and -L to get runtime support
	- `onnx-mlir` can use -l and -L to link external library.

# How do we compile with Posit Wrapper

1. Remember we should use add `LD_LIBRARY_PATH` to path
2. Compile source file to object file
3. Use clang to link the object file with library with -L and -l

- We use `source.c` to compile with wrapper for testing.
- The working command.
	- `export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH`
	- `clang -c -o source.o source.c`
	- `clang -o test.exe source.o -L/custom_posit/lib/ -lposit_c_api_custom`
	- `./test.exe`
- Note: I haven't get include working

---

real command
`clang -c -o test_libposit.o test_libposit.c`
`export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH`
`clang -o test.exe test_libposit.o -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`

---
# Unified way to add operation in wrapper

```c
#define SOURCE_NBITS_ESVAL(bits, es_val)                                       \
  SOURCE_POSIT_BASIC(bits, es_val, add, +)                                     \
  SOURCE_POSIT_BASIC(bits, es_val, sub, -)                                     \
  SOURCE_POSIT_BASIC(bits, es_val, mul, *)                                     \
  SOURCE_POSIT_BASIC(bits, es_val, div, /)                                     \
  SOURCE_POSIT_CMP(bits, es_val, oeq, ==)                                      \
  SOURCE_POSIT_CMP(bits, es_val, ogt, >)                                       \
  SOURCE_POSIT_SELECT(bits, es_val)
```

```c
#define DEFINE_NBITS_ESVAL(bits, es_val)                                       \
  DEFINE_POSIT(bits, es_val, add)                                              \
  DEFINE_POSIT(bits, es_val, sub)                                              \
  DEFINE_POSIT(bits, es_val, mul)                                              \
  DEFINE_POSIT(bits, es_val, div)                                              \
  DEFINE_POSIT_BOOL(bits, es_val, oeq)                                         \
  DEFINE_POSIT_BOOL(bits, es_val, ogt)                                         \
  DEFINE_POSIT_SELECT(bits, es_val)
```

verify:
```cpp
nm libposit_c_api_custom.a | grep "posit.*add"
000000000000be90 T posit16es0_add
000000000000d610 T posit16es1_add
000000000000f7e0 T posit16es2_add
00000000000119e0 T posit16es3_add
0000000000013750 T posit32es0_add
0000000000014e90 T posit32es1_add
0000000000016b70 T posit32es2_add
0000000000018d70 T posit32es3_add
00000000000042d0 T posit8es0_add
0000000000005c30 T posit8es1_add
00000000000079c0 T posit8es2_add
00000000000096e0 T posit8es3_add
```

# Linking to get to end to end

- Now we did the following in wrapper:
	- Compiled file to object file.
	- Get the -l and -L from previous compilation test. 
- Same as we did in `onnx-mlir`, now we try to compile end-to-end.
- Command; (mnist_post.cpp is user driver code)
	- `export LD_LIBRARY_PATH=/custom_posit/lib:$LD_LIBRARY_PATH`
	- `onnx-mlir -EmitLib --enable-posit --n-bits=8 --es-val=2 mnist_model.onnx -o mnist_posit -L/custom_posit/lib/ -lposit_c_api_custom`
	-  `g++ --std=c++11 -O3 mnist_posit.cpp ./mnist_posit.so -o mnist_posit -I $ONNX_MLIR_INCLUDE -L/custom_posit/lib/ -lposit_c_api_custom`
	- `./mnist_posit`

---

`export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH`

`onnx-mlir -EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o mnist_posit -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`

 `g++ --std=c++11 -O3 mnist_posit.cpp ./mnist_posit.so -o mnist_posit -I $ONNX_MLIR_INCLUDE -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`
 ---

# Result

- Execute:
	- We can run but the user driver is not modified.
```cpp
./mnist_posit
prediction[0] = 40309352949036252632845219700670464.000000
prediction[1] = -2479245573310062766748096079593472.000000
prediction[2] = 0.000000
prediction[3] = 0.000000
prediction[4] = 295173594853019636725962506240.000000
prediction[5] = 0.000000
prediction[6] = 0.000000
prediction[7] = 0.000000
prediction[8] = 0.000000
prediction[9] = 0.000000
The digit is 0
```
 
---

# Wait, there's last puzzle we does not solve!

## Notorious `_mlir_ciface_`

both posit and non-posit has ciface
`_mlir_ciface_main_graph_llvm`
`_mlir_ciface_main_graph_llvm_log.txt`

original: 3657kb

posit8: 1276kb
posit16: 2197kb
posit32: 4059kb

trace `omTensorSetDataPtr` generation

```
  using API = RuntimeAPI::API;
  std::vector<RuntimeAPI> RuntimeAPISpecs = {
    RuntimeAPI(API::CREATE_OMTENSOR_LIST, "omTensorListCreate", opaquePtrTy, {opaquePtrPtrTy, int64Ty}),
    RuntimeAPI(API::CREATE_OMTENSOR, "omTensorCreateUntyped", opaquePtrTy, {int64Ty}),
    RuntimeAPI(API::DESTROY_OMTENSOR, "omTensorDestroy", voidTy, {opaquePtrTy}),
    RuntimeAPI(API::GET_DATA, "omTensorGetDataPtr", opaquePtrTy, {opaquePtrTy}),
    RuntimeAPI(API::SET_DATA, "omTensorSetDataPtr", voidTy, {opaquePtrTy, int64Ty, opaquePtrTy, opaquePtrTy}),
```

```
  for (auto &apiSpec : RuntimeAPISpecs) {
    apiSpec.declareAPI(module, builder);
    registry.emplace(apiSpec.id, apiSpec);
  }
```

```
  const RuntimeAPI &getAPI(RuntimeAPI::API apiId) const {
    assert((registry.find(apiId) != registry.end()) &&
           "apiId not found in registry");
    return registry.at(apiId);
  }
```

callApi would generate `llvm.call` with input enums

```
  return create.llvm.call(ArrayRef<Type>(outputTys),
      registry.getAPI(apiId).symbolRef, ArrayRef<Value>(params));
```

declaration:
declare at `RuntimeAPIRegistry::RuntimeAPIRegistry` constructor

`declareAPI`

```
create.llvm.getOrInsertSymbolRef(module, name, outputTy, inputTys);
```

RuntimeAPIRegistry constructor called in `ConvertKrnlToLLVMPass::runOnOperation`

it's only constructed once
constructed at "invoke at KrnlEntryPointOpLowering"

Summary:

declaration is added last.

`krnl::populateKrnlToLLVMConversion` (the last pattern in populateAffineAndKrnlToLLVMConversion)
in ConvertKrnlToLLVMPass

compiler option: `./onnx-mlir --EmitLLVMIR --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o LLVM`

TODO: call location

# Log

```
./onnx-mlir --EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom -v

Onnx-mlir command: 
./onnx-mlir --EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom -v

The ONNX model has 421642 elements in its initializers. This value would be close to and greater than the number of parameters in the model. Because there is no way to exactly count the number of parameters, this value can be used to have a rough idea of the number of parameters in the model.

[/home/sylvex/onnx-mlir/build/Debug/bin/] /home/sylvex/onnx_llvm/llvm-project/build/bin/opt: opt -O0 --mtriple=x86_64-unknown-linux-gnu --code-model small -o model.so.bc model.so.unoptimized.bc                                                                                                                       [/home/sylvex/onnx-mlir/build/Debug/bin/] /home/sylvex/onnx_llvm/llvm-project/build/bin/llc: llc -O0 --mtriple=x86_64-unknown-linux-gnu --code-model small -filetype=obj -relocation-model=pic -o model.so.o model.so.bc                                                                                                [/home/sylvex/onnx-mlir/build/Debug/bin/] /usr/bin/clang++: clang++ model.so.o -o model.so.so -shared -fPIC -L/home/sylvex/onnx-mlir/build/Debug/lib -L/home/sylvex/custom_posit/lib/ -lcruntime -lposit_c_api_custom                                                                                                   Shared library 'model.so.so' has been compiled.
```

```bash
ldd model.so

linux-vdso.so.1 (0x00007fff7a13a000)
libposit_c_api_custom.so => /home/sylvex/custom_posit/lib/libposit_c_api_custom.so (0x00007a975f200000)                                                     libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007a975ee00000)
libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007a975f0f8000)                                                                                           libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007a975f0d8000)
libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007a975ea00000)
/lib64/ld-linux-x86-64.so.2 (0x00007a975f2ba000)
```

execute

```cpp
./mnist_posit
prediction[0] = 40309352949036252632845219700670464.000000
prediction[1] = -2479245573310062766748096079593472.000000
prediction[2] = 0.000000
prediction[3] = 0.000000
prediction[4] = 295173594853019636725962506240.000000
prediction[5] = 0.000000
prediction[6] = 0.000000
prediction[7] = 0.000000
prediction[8] = 0.000000
prediction[9] = 0.000000
The digit is 0
```

# Where the `_mlir_ciface_` generated?

```
// Request C wrapper emission via attribute.
for (auto func : module.getOps<func::FuncOp>()) {
func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
	UnitAttr::get(&getContext()));
}
```

comment and it passed

original ciface still exist

addKrnlToAffinePasses -> Our Pass -> Enter ConvertKrnlToLLVMPass -> Inject C Wrapper Attribute -> Apply ToLLVM conversion