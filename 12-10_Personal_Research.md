
# Summary

- Posit prove of concept work.

# TODO:

- Rewrite the pass
	- Put our pass after the affineToStd
	- Should we just lower the pass to llvm directly?
		- `FuncToLLVM` difference
- Design a experiment
	- Load data set, data transformation
	- Metric of measuring posit precision with different config
	- So far using c++, maybe i need to come up serialize output scheme.
		- json to serialize from python and load json in to c++??
	- How does different model input.
- Write about how do I make experiment work.

- Mnist input: [1 , 1 , 28 , 28]

data transform

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

Typically, `scf` is lowered to `cf` and then lowered to some final target like LLVM or SPIR-V.

`cf.assert`: assert(i1) -> str
```mlir
cf.assert %b, "Expected ... to be true"
```
`cf.br`: br(any)
```mlir
^bb2:
  %2 = call @someFn()
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```
`cf.cond_br`
```mlir
cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
```
`cf.switch`
```mlir
cf.switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```

The last three has `BranchOpInterface`: Can it be lowered by `populateBranchOpInterfaceTypeConversionPattern`?

scf at most has `RegionBranchOpInterface`
scf has `populateSCFStructuralTypeConversionsAndLegality`

affine for and if also has `RegionBranchOpInterface`
no populate ???

Remember we have standard lowering!
```
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
```

`populateReturnOpTypeConversionPattern`
```cpp
rewriter.modifyOpInPlace(op, [&] { op->setOperands(adaptor.getOperands()); });
```

`populateBranchOpInterfaceTypeConversionPattern`
```cpp
  LogicalResult
  matchAndRewrite(BranchOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // For a branch operation, only some operands go to the target blocks, so
    // only rewrite those.
    SmallVector<Value, 4> newOperands(op->operand_begin(), op->operand_end());
    for (int succIdx = 0, succEnd = op->getBlock()->getNumSuccessors();
         succIdx < succEnd; ++succIdx) {
      OperandRange forwardedOperands =
          op.getSuccessorOperands(succIdx).getForwardedOperands();
      if (forwardedOperands.empty())
        continue;

      for (int idx = forwardedOperands.getBeginOperandIndex(),
               eidx = idx + forwardedOperands.size();
           idx < eidx; ++idx) {
        if (!shouldConvertBranchOperand || shouldConvertBranchOperand(op, idx))
          newOperands[idx] = operands[idx];
      }
    }
    rewriter.modifyOpInPlace(
        op, [newOperands, op]() { op->setOperands(newOperands); });
    return success();
  }

```

# Where should our pass locate

@TODO: list the target dialect

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
# Experiment result

```bash
(base) sylvex@sylvex-Aspire-A715-51G:~/mlir_posit$ time ./mnist
prediction[0] = 31.543524
prediction[1] = -19.310675
prediction[2] = -2.363821
prediction[3] = -11.919171
prediction[4] = -0.440006
prediction[5] = -1.079402
prediction[6] = 3.012452
prediction[7] = -1.341520
prediction[8] = -1.788554
prediction[9] = 11.343985
The digit is 0

real    0m0.025s
user    0m0.024s
sys     0m0.001s

(base) sylvex@sylvex-Aspire-A715-51G:~/mlir_posit$ time ./mnist_posit
prediction[0] = 31.543522
prediction[1] = -19.310673
prediction[2] = -2.363823
prediction[3] = -11.919170
prediction[4] = -0.440003
prediction[5] = -1.079404
prediction[6] = 3.012451
prediction[7] = -1.341515
prediction[8] = -1.788554
prediction[9] = 11.343984
The digit is 0

real    0m10.290s
user    0m10.289s
sys     0m0.000s
```
411.6x slow down

# Test Pipeline

[pipeline link](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/)

The test is run through RunONNXModel.py