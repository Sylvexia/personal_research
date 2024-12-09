
# Summary

- Posit proof of concept works.
# TODO:

- Rewrite the pass
	- Put our pass after the `affineToStd`
	- Should we just lower the pass to `llvm` directly?
		- `FuncToLLVM` difference
- Design a experiment
	- Load data set, data transformation.
	- Metric of measuring posit precision with different config.
	- See the how the `onnx-mlir` test doing?
		- So far using `c++`, maybe i need to come up serialize output scheme.
			- `json` to serialize from python and load `json` in to `c++`??
		- How to handle different model input?
- Recent big goal
	- posit dialect
		- `add`, `mul`, `const`
		- `exp`, `sqrt`, 
		- `load`, `store`, `alloc`, `alloca`
		- Our goal would be more like SPIR-V dialect.

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

- What did we lower?
	- `arith.const` and `krnlGlobalOp`
		- Numerical conversion should not be done by pass.
		- Ultimate goal: lowering should not involve project's custom operation like `krnlGlobalOp`
	- `arith` to `func`
		- We still need to rethink should we lower to `llvm.func` directly
		- We mostly don't need `Func` dialect specific feature
			- `func.func` to `llvm.func`
				- convert type
				- handle based on attribute, 
					- linkage, propagate, c wrapper
				- create `llvm.func`
				- inline function body
		- However, there does exist target specific like GPU or SPIRV
			- `populateFuncToSPIRVPatterns()`
	- `affine`
		- It would lower to `scf` then `cf` dialect, we only care about `cf`.
		- SCF to CF, what's the difference?
			- SCF use MLIR region to contain block of operations.
				- if, for, while, parallel...
			- CF just use SSA blocks, think of it as labels.
				- `assert`, `br`, `cond_br`, `switch`
		- https://github.com/j2kun/mlir-tutorial/pull/20/commits/25b284b48cbc18860aac6edff59f5eb6b9466268
	- `memref`
		- `AllocaOp`, `AllocOp`, `LoadOp`, `ReinterpretCastOp`
		- `StoreOp` is done in affine.

- Before the following listing, the `ConvertKrnlToLLVM` pass does the following
	- Deal with entry point lowering
	- Extract Constants to File if enabled
	- Request C wrapper emission via attribute.
	1. **Append Postfix to Entry Points**: Adds a unique string from the module's attribute `onnx-mlir.symbol-postfix` to each entry point function name.
	2. **Initialize Entry Point ID**: Sets `KRNL_ENTRY_POINT_ID` to 0.
	3. **Prepare Global Ops**: Initializes vectors to store global operations for entry point names and their input/output JSON signatures.
	4. **Record Original MemRefTypes**: Records the original `MemRefType` for inputs and outputs before they are lowered to LLVM IR.
	5. **Check Single Entry Point**: Determines if the module has exactly one entry point.
	6. **Determine OMTensor Ownership**: Determines whether each output `OMTensor` should own its underlying buffer.
	7. **Extract Constants to File**: If enabled, extracts constants from the module and writes them to a binary file if they meet size thresholds.
	8. Request C wrapper emission via attribute.

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

| posit config | runtime (seconds) | match label |
| ------------ | ----------------- | ----------- |
| F32          | 0.028482          | Yes         |
| Posit(32, 0) | 11.464            | Yes         |
| Posit(32, 1) | 11.182            | Yes         |
| Posit(32, 2) | 10.546            | Yes         |
| Posit(32, 3) | 10.713            | Yes         |
| Posit(16, 0) | 2.570             | No (Nan)    |
| Posit(16, 1) | 2.538             | Yes         |
| Posit(16, 2) | 2.476             | Yes         |
| Posit(16, 3) | 2.379             | Yes         |
| Posit(8, 0)  | 0.379             | No (Nan)    |
| Posit(8, 1)  | 0.532             | No (Nan)    |
| Posit(8, 2)  | 0.500             | Yes         |
| Posit(8, 3)  | 0.420             | Yes         |

run_main_graph took 0.028482 seconds
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

```bash
posit config: (8, 0):
run_main_graph took 0.378976 seconds
prediction[0] = nan
prediction[1] = nan
prediction[2] = nan
prediction[3] = nan
prediction[4] = nan
prediction[5] = nan
prediction[6] = nan
prediction[7] = nan
prediction[8] = nan
prediction[9] = nan
The digit is -1
posit config: (8, 1):
run_main_graph took 0.531552 seconds
prediction[0] = 0.019531
prediction[1] = 0.101562
prediction[2] = -0.046875
prediction[3] = 0.015625
prediction[4] = -0.109375
prediction[5] = 0.046875
prediction[6] = -0.109375
prediction[7] = -0.062500
prediction[8] = 0.062500
prediction[9] = -0.027344
The digit is 1
posit config: (8, 2):
run_main_graph took 0.499995 seconds
prediction[0] = 15.000000
prediction[1] = -13.000000
prediction[2] = -0.687500
prediction[3] = -3.500000
prediction[4] = -3.500000
prediction[5] = -1.625000
prediction[6] = 2.500000
prediction[7] = -2.500000
prediction[8] = 1.750000
prediction[9] = 6.500000
The digit is 0
posit config: (8, 3):
run_main_graph took 0.420289 seconds
prediction[0] = 8.000000
prediction[1] = -4.000000
prediction[2] = -0.375000
prediction[3] = -4.000000
prediction[4] = -0.500000
prediction[5] = -0.312500
prediction[6] = 0.625000
prediction[7] = -0.750000
prediction[8] = 0.156250
prediction[9] = 3.000000
The digit is 0
posit config: (16, 0):
run_main_graph took 2.570433 seconds
prediction[0] = -0.164795
prediction[1] = -0.271057
prediction[2] = -0.061523
prediction[3] = -1.299438
prediction[4] = 1.395264
prediction[5] = 0.595581
prediction[6] = nan
prediction[7] = -0.316284
prediction[8] = nan
prediction[9] = 1.254883
The digit is 4
posit config: (16, 1):
run_main_graph took 2.538182 seconds
prediction[0] = 31.515625
prediction[1] = -19.312500
prediction[2] = -2.394043
prediction[3] = -11.925781
prediction[4] = -0.419312
prediction[5] = -1.130859
prediction[6] = 3.031738
prediction[7] = -1.309570
prediction[8] = -1.804932
prediction[9] = 11.335938
The digit is 0
posit config: (16, 2):
run_main_graph took 2.476076 seconds
prediction[0] = 31.484375
prediction[1] = -19.296875
prediction[2] = -2.394531
prediction[3] = -11.917969
prediction[4] = -0.423340
prediction[5] = -1.119629
prediction[6] = 3.041016
prediction[7] = -1.311035
prediction[8] = -1.816406
prediction[9] = 11.351562
The digit is 0
posit config: (16, 3):
run_main_graph took 2.379381 seconds
prediction[0] = 31.484375
prediction[1] = -19.312500
prediction[2] = -2.423828
prediction[3] = -11.882812
prediction[4] = -0.400635
prediction[5] = -1.104492
prediction[6] = 2.978516
prediction[7] = -1.324219
prediction[8] = -1.806641
prediction[9] = 11.421875
The digit is 0
posit config: (32, 0):
run_main_graph took 11.464221 seconds
prediction[0] = 31.543522
prediction[1] = -19.310673
prediction[2] = -2.363822
prediction[3] = -11.919169
prediction[4] = -0.440003
prediction[5] = -1.079404
prediction[6] = 3.012450
prediction[7] = -1.341515
prediction[8] = -1.788554
prediction[9] = 11.343983
The digit is 0
posit config: (32, 1):
run_main_graph took 11.181827 seconds
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
posit config: (32, 2):
run_main_graph took 10.545715 seconds
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
posit config: (32, 3):
run_main_graph took 10.713363 seconds
prediction[0] = 31.543522
prediction[1] = -19.310673
prediction[2] = -2.363821
prediction[3] = -11.919169
prediction[4] = -0.440004
prediction[5] = -1.079404
prediction[6] = 3.012450
prediction[7] = -1.341516
prediction[8] = -1.788554
prediction[9] = 11.343983
The digit is 0
```
# Model to lower

- Mnist input: [1 , 1 , 28 , 28]

data transform

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
- Current model:
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

```
- `whisper`
	- arith
		- sitofp
			- Cast from a value interpreted as a signed integer to the corresponding floating-point value
	- Math
		- exp, sqrt
		- erf
			- $\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}  dt$
- `gpt-2`
	- arith
		- sitofp
	- Math
		- exp, sqrt, tanh

# What we've done to make it work

- MNIST Example in the project  would not work!
	- Our version does not have `softmax` operator, which would not have exp operation.
- Comment out the c interface injection to function declaration in original pass.
	- Still waiting for response in git issue.
- Our pass should not deal with affine. We should just deal with control flow type.
	- Like what we did for the function.
	- `populateBranchOpInterfaceTypeConversionPattern`
- The input value distribution does not fit in current (currently data not normalized)

# Test Pipeline

[pipeline link](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/)

The test is run through RunONNXModel.py

which is called by RunONNXModelZoo.py