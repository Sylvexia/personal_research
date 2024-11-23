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

# How ONNX runtime works?

`compileModuleToSharedLibrary`
-> `compileModuleToObject`

`compileModuleToObject`: 
`genLLVMBitcode` -> `genModelObject`

`genLLVMBitcode`:

`genModelObject`