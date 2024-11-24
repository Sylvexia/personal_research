# Macro-ed wrapper
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

verified with `nm` that has simple symbol name:

`nm c_api/custom/posit/libposit_c_api_custom.a | grep 16`
`00000000000021c0 T posit16es1_add`

`.a` file is static library

# Adding our Pass

What was working:
`./onnx-mlir --EmitMLIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`

Not working
```
./onnx-mlir --EmitLLVMIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o ./llvm_log.txt
loc("onnx.Constant"("Initializer_fc2.bias")): error: redefinition of symbol named 'name_llvm_log.txt'
```
# How ONNX runtime works?

`compileModuleToSharedLibrary`
-> `compileModuleToObject`

`compileModuleToObject`: 
`genLLVMBitcode` -> `genModelObject`

`genLLVMBitcode`:

`genModelObject`

# 

`./onnx-mlir --EmitLLVMIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx --mlir-print-ir-after-failure --mlir-elide-elementsattrs-if-larger=16 -o ./llvm_log.txt`

# Compile Failed

```
FAILED: docs/doc_example/OMRuntimeTest
: && /usr/bin/clang -fPIC -fno-semantic-interposition -Werror=date-time -Werror=unguarded-availability-new -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wmissing-field-initializers -Wimplicit-fallthrough -Wcovered-switch-default -Wstring-conversion -Wmisleading-indentation -Wctad-maybe-unsupported -fdiagnostics-color -DSUPPRESS_THIRD_PARTY_WARNINGS -g  docs/doc_example/CMakeFiles/OMRuntimeTest.dir/main.c.o -o docs/doc_example/OMRuntimeTest -L/home/sylvex/onnx-mlir/build/docs/doc_example -Wl,-rpath,/home/sylvex/onnx-mlir/build/docs/doc_example  -ladd && :
/usr/bin/ld: /home/sylvex/onnx-mlir/build/docs/doc_example/libadd.so: undefined reference to `_mlir_ciface_posit8es8_add'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```

## What is ciface?

In `ConvertKrnlToLLVM.cpp`
`krnlEntryPointOpLowering`
```
// 3. Emit code to prepare MemRefs from OMTensor inputs and call
// `_mlir_ciface` prefixed function of the entry point.
```