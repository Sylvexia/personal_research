# TODO

- Move pass after `cf` dialect.
- Math Dialect lowering.
- How does runtime work?
	- Test pipeline.
- Posit Dialect.
- What does vector Dialect do
```cpp
  // vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  // vector::populateVectorBroadcastLoweringPatterns(patterns);
  // vector::populateVectorContractLoweringPatterns(
  //     patterns, vector::VectorTransformsOptions());
  // vector::populateVectorTransposeLoweringPatterns(
  //     patterns, vector::VectorTransformsOptions());
```
# Summary


# Lowering

```cpp
  RewritePatternSet prePatterns(&getContext());

  populateAffineToStdConversionPatterns(prePatterns);
  populateSCFToControlFlowConversionPatterns(prePatterns);

  ConversionTarget preTarget(getContext());
  preTarget.addIllegalDialect<affine::AffineDialect, scf::SCFDialect>();
  preTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(module, preTarget, std::move(prePatterns))))
    signalPassFailure();

```

TODO: write about the full and partial

## WTF

Doing standard with two `applyPartialConversion` seemed to mixed up.

`./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=3' /home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir --mlir-elide-elementsattrs-if-larger=16 > lowered.mlir`

```cpp
/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:16:12: error: failed to materialize conversion for result #0 of operation 'arith.constant' that remained live after conversion
    %cst = arith.constant 0xFF800000 : f32
           ^
/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:16:12: note: see current operation: %1 = "arith.constant"() <{value = 0xFF800000 : f32}> : () -> f32
/home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir:151:13: note: see existing live user here: "memref.store"(%1, %364) <{nontemporal = false}> : (f32, memref<f32>) -> ()
            affine.store %cst, %alloca_6[] : memref<f32>
            ^
```

# Cannot lower to cf

error: `Only structured control-flow loops are supported.`

before cf, there's bufferDeallocation pass:



# Solution

`scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);`

- Meaning: We do not need to lower control flow related operation 1 by 1.
# Runtime

[pipeline link](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/)

The output from the link is run through RunONNXModel.py

which is called by RunONNXModelZoo.py

## RunONNXModelZoo.py

`ok, msg = execute_commands(RUN_ONNX_MODEL_CMD + options, tmout=1800)`

options = compile_args + data_set + model

## RunONNXModel.py

clone github and download dataset and model.
# Metric of GPT-2

- Perplexity
	- how well a probability model predicts a sample and is commonly used to evaluate language models, lower the better.
- BLEU Score
	- Measures the similarity between the generated text and reference text, often used in machine translation
- ROUGE Score
	- Evaluates the overlap between the generated text and reference summaries, useful for summarization tasks

# Multiple apply?

```cpp
vector::populateVectorToVectorCanonicalizationPatterns(patterns);
vector::populateVectorBroadcastLoweringPatterns(patterns);
vector::populateVectorContractLoweringPatterns(
  patterns, vector::VectorTransformsOptions());
vector::populateVectorTransposeLoweringPatterns(
  patterns, vector::VectorTransformsOptions());

populateAffineToStdConversionPatterns(patterns);
populateSCFToControlFlowConversionPatterns(patterns);
```

# preserveLLVMIR

`--preserveLLVMIR`

```cpp
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@constant_8_mnist_posit = internal constant [1280 x i8] c"*#\9A*)\A3(\A6\9E\9B\B1\1B\10\A2\14\96-\1B\A1\A2\9C\9F*\9B\96&...

declare i32 @strncmp(ptr, ptr, i64)

declare void @omGetExternalConstantAddr(ptr, ptr, i64)

; Function Attrs: memory(none)
declare i1 @posit8es2_ogt(i8, i8) #0

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main_graph_mnist_posit(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10) {
  %12 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (i8, ptr null, i32 25088) to i64), i64 16))

93:                                               ; preds = %40
  %94 = getelementptr i8, ptr @constant_1_mnist_posit, i64 %31
  %95 = load i8, ptr %94, align 1
  %96 = call i8 @posit8es2_add(i8 %42, i8 %95)

define ptr @run_main_graph(ptr %0) {
  %2 = call ptr @run_main_graph_mnist_posit(ptr %0)
  ret ptr %2
}
```