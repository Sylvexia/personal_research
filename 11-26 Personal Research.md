# Summary

- In posit wrapper, using token pasting for `n_bits, es_val` operation.
- In `onnx-mlir`
	- Add our custom pass to main compiler, with input argument.
	- Successfully lowered to `.so`, but there's issue.
		- Fixed the `krnl.globalOp` name duplication. (dedicated for model weight)
		- New issue:
			- Generated function declaration is altered later in `FuncToLLVM` pass
		- Function declaration generation needs to be revised.
# Macro-ed wrapper (Proof of Concept)

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

- using token pasting `##` in c

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

- verified with `nm` that has simple symbol name:

```
nm c_api/custom/posit/libposit_c_api_custom.a | grep 16
00000000000021c0 T posit16es1_add
```

# Adding our Pass

- Pass Order:
	- `ONNXToMLIR` $\rightarrow$ `ONNXToKrnl` $\rightarrow$ `KrnlToAffine` $\rightarrow$ `KrnlToLLVM`
		1. `Represent ONNX in MLIR
		2. Decompose ONNX with custom representation.
		3. Custom loop representation to affine.
		4. Populate standard conversion with additional `Krnl` to LLVM.
- The pass is added immediately after the `KrnlToAffine`
- Two main compiler:
	- `onnx-mlir-opt`: For testing pass separately.
	- `onnx-mlir`: Main compiler, putting all pass together that we can end to end compile.
- Basically for past month, we run our pass `onnx-mlir-opt` and test separately
- Now we can (kind of) execute, compile from `onnx` to `mlir`
	- `./onnx-mlir --EmitMLIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx`

What was working:
`./onnx-mlir --EmitMLIR --n-bits=16 --es-val=2 /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`

`./onnx-mlir --EmitLib /home/sylvex/mnist_export/mnist_model.onnx -o LIB --mlir-elide-elementsattrs-if-larger=16 --mlir-elide-resource-strings-if-larger=16 -mlir-print-stacktrace-on-diagnostic --mlir-print-ir-after-failure 2> LIB_LOG`


# Compilation process in the main compiler

`compileModuleToSharedLibrary`
- `compileModuleToObject`
	- `genLLVMBitcode`:
		- `translateModuleToLLVMIR`: translate MLIR to LLVMIR
		- `tailorLLVMIR`: add thing that not in MLIR but LLVM
		- `WriteBitcodeToFile`: 
		- using `opt` to optimize bitcode
	- `genModelObject`: using `llc` to compile LLVM bitcode to object file.
- `genSharedLib`: using `cxx` to compile and link e.g. `cruntime`, `jniruntime`

link:
```cpp
#define CCM_SHARED_LIB_DEPS "sharedLibDeps"
#define CCM_SHARED_LIB_PATH_DEPS "sharedLibPathDeps"
```

```
addCompilerConfig(CCM_SHARED_LIB_DEPS,
        emissionTarget == EmitLib
            ? std::vector<std::string>{"cruntime"}
            : std::vector<std::string>{"jniruntime", "cruntime"});
addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, {getLibraryPath()});

addCompilerConfig(CCM_SHARED_LIB_DEPS, extraLibs);
addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, extraLibPaths);
```

```cpp
static llvm::cl::list<std::string, std::vector<std::string>> extraLibPathsOpt(
    "L",
    llvm::cl::desc("Specify extra directories for libraries when compiling"
                   "an onnx model. Will be add used as -L in the linkage step."
                   "Each directory can be specified with one extra-lib-dirs"),
    llvm::cl::location(extraLibPaths), llvm::cl::Prefix,
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::list<std::string, std::vector<std::string>> extraLibsOpt("l",
    llvm::cl::desc("Specify extra libraries when compiling an onnx model."
                   "Will be add used as -l in the linkage step."
                   "Each lib can be specified with one extra-libs"),
    llvm::cl::location(extraLibs), llvm::cl::Prefix,
    llvm::cl::cat(OnnxMlirOptions));
```

# Test Failed (ciface issue)

- This is message from building test.
```cpp
/usr/bin/ld: /home/sylvex/onnx-mlir/build/docs/doc_example/libadd.so: undefined reference to `_mlir_ciface_posit8es8_add'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```
- It make me look into the compiled shared library and MLIR file
- `nm LIB.so`:
	```cpp
	00000000000039b0 T _mlir_ciface_main_graph_lib
                 U _mlir_ciface_posit8es8_add
                 U _mlir_ciface_posit8es8_mul
                 U _mlir_ciface_posit8es8_oge
                 U _mlir_ciface_posit8es8_ogt
                 U _mlir_ciface_posit8es8_select
	```
- MLIR:
```cpp
llvm.func private @posit8es8_mul(%arg0: i8, %arg1: i8) -> i8 attributes {llvm.emit_c_interface, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_posit8es8_mul(%arg0, %arg1) : (i8, i8) -> i8
    llvm.return %0 : i8
  }
  llvm.func @_mlir_ciface_posit8es8_mul(i8, i8) -> i8 attributes {llvm.emit_c_interface, sym_visibility = "private"}

```
- It raises a question: what is ciface?


```
FAILED: docs/doc_example/OMRuntimeTest
: && /usr/bin/clang -fPIC -fno-semantic-interposition -Werror=date-time -Werror=unguarded-availability-new -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wmissing-field-initializers -Wimplicit-fallthrough -Wcovered-switch-default -Wstring-conversion -Wmisleading-indentation -Wctad-maybe-unsupported -fdiagnostics-color -DSUPPRESS_THIRD_PARTY_WARNINGS -g  docs/doc_example/CMakeFiles/OMRuntimeTest.dir/main.c.o -o docs/doc_example/OMRuntimeTest -L/home/sylvex/onnx-mlir/build/docs/doc_example -Wl,-rpath,/home/sylvex/onnx-mlir/build/docs/doc_example  -ladd && :
/usr/bin/ld: /home/sylvex/onnx-mlir/build/docs/doc_example/libadd.so: undefined reference to `_mlir_ciface_posit8es8_add'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```

## What is ciface?

- `ONNXToMLIR` -> `ONNXToKrnl` -> `KrnlToAffine` -> `KrnlToLLVM`
	- Where does the `_ciface_` get added?
	- Not in `KrnlToAffine`
	- In `KrnlToLLVM`, I suspect that the pass "`FuncToLLVM`" add `_mlir_ciface_` prefix to function declaration symbol.
- Clue:
	- See runtime function `@omTensorGetDataPtr` generation, it does not have `_mlir_ciface_`
- Future path:
	- Revise our function generating scheme.
	- Move our pass after `_mlir_ciface_` generation.

Called in `populateAffineAndKrnlToLLVMConversion`

listing:
```cpp
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  memref::populateExpandOpsPatterns(patterns);
  // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
  // performance.
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // Enable OpenMP-to-LLVM pass when enable parallelism
  if (enableParallel) {
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  }
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, inputMemRefTypes, outputMemRefTypes, verifyInputTensors);

```

In `FuncToLLVM.cpp`
```cpp
/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. This function can be called from C
/// by passing a pointer to a C struct corresponding to a memref descriptor.
/// Similarly, returned memrefs are passed via pointers to a C struct that is
/// passed as additional argument.
/// Internally, the auxiliary function unpacks the descriptor into individual
/// components and forwards them to `newFuncOp` and forwards the results to
/// the extra arguments.
static void wrapForExternalCallers(...)
auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
	loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
	wrapperFuncType, LLVM::Linkage::External, /*dsoLocal=*/false,
	/*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
```

# Conclusion:

- 