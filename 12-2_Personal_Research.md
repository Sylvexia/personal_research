
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

callApi would generate `llvm.call`

```
  return create.llvm.call(ArrayRef<Type>(outputTys),
      registry.getAPI(apiId).symbolRef, ArrayRef<Value>(params));
```

- omGetExternalConstantAddr not getting called
- omMMapBinaryFile
- omTensorListGetSize
- omTensorPrint