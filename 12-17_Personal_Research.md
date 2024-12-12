# TODO

- Move pass after `cf` dialect.
- Math Dialect lowering.
- How does runtime work?
	- Test pipeline.
- Posit Dialect.
- Test `memref` store
- How to log out the module
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