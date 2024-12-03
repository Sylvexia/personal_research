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

# 12-3 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Unified way to add operation in posit wrapper
	- Supported all posit config of $+$, $-$, $\times$, $\div$, select, compare
	- Verified how to link the library with custom c example.
- Trying to work with end-to-end flow
	- Linked the wrapper to the MNIST model
	- Compiled with user driver code.
	- We can execute and get the model result.
	- Result is currently not verified.

---

# How to compile model to library

1. Set Environment variable
2. Using `onnx-mlir`, to compile`model.onnx` -> `model.so`
3. Using `g++`, to compile `userDriver.cpp` + `model.so` -> `run.exe`
4. You can execute and get the result from `run.exe`

- command:

1. `export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH`
2. `onnx-mlir -EmitLib mnist.onnx`
3. `g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE`
4. `./mnist`

---

# What should do to include posit?

- 2 compiler, `onnx-mlir` and `g++`, for linking
- Intuitively, we should look into what `onnx-mlir` do
	1. MLIR -> LLVM
	2. LLVM -> `.bc`
	3. `.bc` -> `.o` using `llc`
	4. `.o` -> `.so` using `clang++`

---

# What should do to include posit?

- We can log and get the `llc` and `clang++` command with -v in `onnx-mlir` (only important flag is shown here)
	- `llc -filetype=obj -o model.o model.bc`
	- `clang++ model.o -o model.so -shared -L/home/sylvex/onnx-mlir/build/Debug/lib -lcruntime`
- Clue: 
	- Using -l and -L to get runtime support
	- `onnx-mlir` can use -l and -L to link external library.

---

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

---

# Unified way to add operation in wrapper

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

---

# Linking to get to end to end

- Now we did the following in wrapper:
	- Compiled file to object file.
	- Get the -l and -L from previous compilation test. 
- Same as we did in `onnx-mlir`, now we try to compile end-to-end.

---

# Linking to get to end to end

- Command: (mnist_post.cpp is user driver code)
	- `export LD_LIBRARY_PATH=/custom_posit/lib:$LD_LIBRARY_PATH`
	- `onnx-mlir -EmitLib --enable-posit --n-bits=8 --es-val=2 mnist_model.onnx -o mnist_posit -L/custom_posit/lib/ -lposit_c_api_custom`
	-  `g++ --std=c++11 -O3 mnist_posit.cpp ./mnist_posit.so -o mnist_posit -I $ONNX_MLIR_INCLUDE -L/custom_posit/lib/ -lposit_c_api_custom`
	- `./mnist_posit`

---

# Result

- Execute:
	- We can run but the user driver code is not modified.
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

# User Driver Code:

```cpp
extern "C" OMTensorList *run_main_graph(OMTensorList *);
static float img_data[] = {-0.4242129623889923f, -0.4242129623889923f...}

int main() {
  int inputNum = 1;
  OMTensor *inputTensors[inputNum];
  int64_t rank = 4;
  int64_t shape[] = {1, 1, 28, 28};
  OMTensor *tensor = omTensorCreate(img_data, shape, rank, ONNX_TYPE_FLOAT);
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);
  OMTensorList *tensorListOut = run_main_graph(tensorListIn);
  omTensorListDestroy(tensorListIn);

  OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *prediction = (float *)omTensorGetDataPtr(y);
```

---

# User Driver Code:

```cpp
  int digit = -1;
  float prob = 0.;
  for (int i = 0; i < 10; i++) {
    printf("prediction[%d] = %f\n", i, prediction[i]);
    if (prediction[i] > prob) {
      digit = i;
      prob = prediction[i];
    }
  }

  // Free the output as it is no longer needed.
  omTensorListDestroy(tensorListOut);

  printf("The digit is %d\n", digit);
  return 0;
}
```

---

# User Driver Code:

- Declare `*run_main_graph` as inference entry point, given input and spit out output
- `OMTensorList` is a batch of data, consist of `OMTensor`
- `OMTensor` is a singular data, say all image pixel.
- using `omTensorCreate`, `omTensorListCreate` to help create `OMTensor(List)`
- `omTensorListGetOmtByIndex`, `omTensorGetDataPtr` to get the data from pointer

---

# Wait, there's a last puzzle we do not solve!

## Notorious `_mlir_ciface_`

---

# Solution

- Comment and it passed
```
// Request C wrapper emission via attribute.
for (auto func : module.getOps<func::FuncOp>()) {
func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
	UnitAttr::get(&getContext()));
}
```
- It add the `llvm.emit_c_interface`, later `FuncToLLVM()` lowering, generate the `_mlir_ciface_` declaration.
- There's still other `_mlir_ciface_main_graph_llvm`, which exist for posit and non posit, haven't investigate further.

---

# What should be our approach?

- `addKrnlToAffinePasses` -> Our Pass -> Enter `ConvertKrnlToLLVMPass` -> Inject C Wrapper Attribute -> Apply `ToLLVM` conversion
- 2 Methods
	- Directly lower the function to LLVM dialect instead of Func dialect.
	- Move our pass after the C Wrapper injection.
- Things to consider:
	- Minimum requirement:
		- Map math operation and arithmetic to posit.
		- Load/Store type.
		- Should we care about control flow?

---

# Future Work

- Verify our approach is numerically correct
- Once our approach is numerically correct, we can start to separate out our interface slowly.