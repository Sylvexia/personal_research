
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
`clang -o test.exe test_libposit.o -L/home/sylvex/custom_posit/lib/ -lposit_c_api_custom`
we also need to `export LD_LIBRARY_PATH=/home/sylvex/custom_posit/lib:$LD_LIBRARY_PATH
`