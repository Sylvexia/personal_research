
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

1. 


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

except from the include
now we can compile with following command
`clang -c -o test_libposit.o test_libposit.c`
we also need to `export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH`
`clang -o test.exe test_libposit.o -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`


```
./onnx-mlir --EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom -v
Onnx-mlir command: ./onnx-mlir --EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom -v
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

Assume env variable is properly set up

```
export ONNX_MLIR_ROOT=$(pwd)/../..
export ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin
export ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include
export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH
export ONNX_MLIR_RUNTIME_DIR=../../build/Debug/lib
```
`export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH`

`onnx-mlir -EmitLib mnist.onnx`
`g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE`
`./mnist`

`onnx-mlir -EmitLib --enable-posit --n-bits=8 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o mnist_posit -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`

generate mnist_posit.so

 `g++ --std=c++11 -O3 mnist_posit.cpp ./mnist_posit.so -o mnist_posit -I $ONNX_MLIR_INCLUDE -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`
 
```
/usr/bin/ld: ./mnist_posit.so: undefined reference to `_mlir_ciface_posit8es2_select'
/usr/bin/ld: ./mnist_posit.so: undefined reference to `_mlir_ciface_posit8es2_mul'
/usr/bin/ld: ./mnist_posit.so: undefined reference to `_mlir_ciface_posit8es2_oge'
/usr/bin/ld: ./mnist_posit.so: undefined reference to `_mlir_ciface_posit8es2_ogt'
/usr/bin/ld: ./mnist_posit.so: undefined reference to `_mlir_ciface_posit8es2_add'
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