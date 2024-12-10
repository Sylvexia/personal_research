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
# 12-10 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Posit proof of concept works.

---

# How does it works?

- Singular project for experiment, just need to know the library path.
- Copy and paste my posit-uint code.
- Include the universal header only library.
- Many macros in driver code and environment variable in shell script , so you can
	- `-DN_BIT_VAR=$N_BITS -DES_VAR=$ES_VAL` as compiler directive to feed the variable to user driver code and compile.
	- Use environment variable to config `n_bit`, `es_val`, linking directory location...

---

# User driver code modification:

Wrap to tensor:
```cpp
std::vector<RAW_DATA_TYPE> posit_data;
for (float val : img_data) {
	sw::universal::posit<N_BIT, ES> posit(val);
	auto posit_bit = get_uType<N_BIT, ES, RAW_DATA_TYPE>(posit);
	posit_data.push_back(posit_bit);
}
OMTensor *tensor =
	omTensorCreate(posit_data.data(), shape, rank, ONNX_MAP_TYPE);
```

---

# User driver code modification:

Unwrap from tensor
```cpp
RAW_DATA_TYPE *prediction = (RAW_DATA_TYPE *)omTensorGetDataPtr(y);
// ...
auto pa = get_posit<N_BIT, ES>(prediction[i]);
float floatVal = static_cast<float>(pa);
```

---

# Ground Truth Log

```
Unchanged
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
```
---

# Posit (32, 2) Log

```
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
```

---

# Posit (8, 2) Log

```
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
```
---

# Posit (16, 0) Log

```
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
```

---

# Experiment result

| posit config | runtime (seconds) | match label |
| ------------ | ----------------- | ----------- |
| F32          | 0.028482          | Yes         |
| Posit(32, 0) | 11.464            | Yes         |
| Posit(32, 1) | 11.182            | Yes         |
| Posit(32, 2) | 10.546            | Yes         |
| Posit(32, 3) | 10.713            | Yes         |

---

# Experiment result

| posit config | runtime (seconds) | match label |
| ------------ | ----------------- | ----------- |
| Posit(16, 0) | 2.570             | No (Nan)    |
| Posit(16, 1) | 2.538             | Yes         |
| Posit(16, 2) | 2.476             | Yes         |
| Posit(16, 3) | 2.379             | Yes         |

---

# Experiment result

| posit config | runtime (seconds) | match label |
| ------------ | ----------------- | ----------- |
| Posit(8, 0)  | 0.379             | No (Nan)    |
| Posit(8, 1)  | 0.532             | No (Nan)    |
| Posit(8, 2)  | 0.500             | Yes         |
| Posit(8, 3)  | 0.420             | Yes         |

---

# Experiment result

| posit config | runtime (seconds) | match label |
| ------------ | ----------------- | ----------- |
| F32          | 0.028482          | Yes         |
| Posit(8, 2)  | 0.500             | Yes         |
| Posit(16, 2) | 2.476             | Yes         |
| Posit(32, 2) | 10.546            | Yes         |

---

# Small Summary

- Posit operartion is $O(bitwidth^2)$ ??
- Posit(32, 2) vs. F32 is 376.42x slow down
- Similar bitwidth has similar runtime under different es-bit
- Match label doesn't work may because the input data is not normalized

---

# Model

- Mnist input: [1 , 1 , 28 , 28]
- Data Transform `transforms.Normalize((0.5,), (0.5,))`
  - Normalize with mean and std is 0.5
- Model Architecture
  - Consist of Conv2d, MaxPool2d, Linear, and ReLU

---

# Model

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

---

# What we've done to make it work

- MNIST Example in the project  would not work!
	- Our version does not have `softmax` operator, which would not have exp operation.
- Comment out the c interface injection to function declaration in original pass.
	- Still waiting for response in git issue.
- Our pass should not deal with affine. We should just deal with control flow type.
	- Like what we did for the function.
	- `populateBranchOpInterfaceTypeConversionPattern`
- The input value distribution does not fit in current model
	- Currently data not normalized

---

# Operation Needed to support for other models

- `whisper`
  - sitofp
	- Cast from a value interpreted as a signed integer to the corresponding floating-point value
  - exp, sqrt
  - erf
	- $\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}  dt$
- `gpt-2`
  - arith
    - sitofp
  - Math
	- exp, sqrt, tanh

---

# Where should our pass locate

- What did we lower?
	- `arith.const` and `krnlGlobalOp`
		- Numerical conversion should not be done by pass.
		- Ultimate goal: lowering should not involve project's custom operation like `krnlGlobalOp`
  
---

# Where should our pass locate

- What did we lower?
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

---

# Where should our pass locate

- What did we lower?
  - `affine`
    - It would lower to `scf` then `cf` dialect, we only care about `cf`.
    - SCF to CF, what's the difference?
  	  - SCF use MLIR region to contain block of operations.
  	    - if, for, while, parallel...
  	  - CF just use SSA blocks, think of it as labels.
  		- `assert`, `br`, `cond_br`, `switch`
    - CF has existing: `populateBranchOpInterfaceTypeConversionPattern`
  	  - Just like we convert existing function type :`populateFunctionOpInterfaceTypeConversionPattern`
  	  - Also like [here:](https://github.com/j2kun/mlir-tutorial/pull/20/commits/25b284b48cbc18860aac6edff59f5eb6b9466268)

---

# Where should our pass locate

- What did we lower?
  - `memref`
	  - `AllocaOp`, `AllocOp`, `LoadOp`, `ReinterpretCastOp`
  - `StoreOp` is done in affine.

---

# Where should our pass locate

- Before the following listing lowering pass, the `ConvertKrnlToLLVM` pass does the following
	- Deal with entry point lowering
	- Extract Constants to File if enabled
	- Request C wrapper emission via attribute.

---

# Where should our pass locate

Listing:

```cpp
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  memref::populateExpandOpsPatterns(patterns);
```

---

# Where should our pass locate

```cpp
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

  if (enableParallel) {
    populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);
  }
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, inputMemRefTypes, outputMemRefTypes, verifyInputTensors);
```

---

# Future Works

- Add more operator support
  - Math and Memref operation lowering.
- Fix the pass
	- Put our pass after the `scfTocf`
	- Rethink we just lower the pass to `llvm` directly?
- Design a experiment
	- Load data set, data transformation.
	- Metric of measuring posit precision with different config.
	- See the how the `onnx-mlir` test doing?
		- So far experiment use `c++`, with macro.
		- How to handle different model input?

---

# Future Works
- Recent big goal
	- Posit dialect
		- `add`, `mul`, `const`
		- `exp`, `sqrt`, 
		- `load`, `store`, `alloc`, `alloca`
		- Our goal would be more like SPIR-V dialect.