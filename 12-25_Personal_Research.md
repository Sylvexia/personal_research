
# TODO

- Runtime
- Custom attribute.
- Math Dialect lowering.
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
# Done

- C and Python Wrapper proof of concept
	- Separated universal wrapper. No longer inside universal library
	- Universal as 3rd party be cloned by submodule
	- `f32` and `uint` conversions in posit in python version for testing pipeline

# Flow Chart

![](note_image/Pasted%20image%2020241225013150.png)

# Answer to Posit Library

1. What library should have
	1. Currently it only support various config of posit
	2. Should we full support of posit config?
		1. Most advanced posit arithmetic has `numpy` derived from `softposit`
			1. (8, 0), (16, 1), (32, 2)
		2. Heard universal has python wrapper but can't found one
2. Unified test to verify the correctness.
3. Auto compile and install
4. Documentation

2, 3, 4 currently has prototype.
# Where does the code come from??
# Issue of the code

- The code basically copy of the implementation `fdlibm` in `netlib`
	- License of the `netlib`
- Only support 64-bit es 2, 3, 4
- Lower bit of operation have high chance not work with current implementation.
	- What approximation scheme is universal? Horner's approximation?
- Some operation does not work e.g. exp

```cpp
Posit64 y, hi, lo, c, t;
__int32_t k = 0,xsb;
__uint32_t hx;

if(x>1) return exp(x-1) * exp(1);
else if(x<-1) return exp(x+1) / exp(1);

GET_HIGH_WORD(hx,x);
```

# Contribution

- An MLIR-based compilation flow to support the evaluation of posit numbers in neural network model inference.
- Experiment: evaluate at least 2 or 3 NN models.
	- Numerical Error and E2E
	- If we can some operation in math, we might be the first team run gpt-2 with posit.
- A detail description related to the solutions for using posit numbers in MLIR, including their advantages and disadvantages.
# Test Pipeline Current Status

- What we had (not done by us):
	- Specifying the model and automatically pull from `github`
		- Both model and dataset.
	- Automatically compile and run with python runtime.
	- Able to feed the compile flag.
- Currently ongoing
	- Python Wrapper (just completed)
	- Bring the custom wrapper to compile command line.


```python
parser.add_argument(
	"-m",
	"--model",
	metavar="model_name",
	help="Only process a list of models in the ONNX model zoo."
	" Passing the name of the models, e.g. 'mnist-8 yolov4'."
	" Use -p to know model names. Without -m, the script "
	" checks all models in the model zoo.",
)
```

```bash
cd build/
source env/bin/activate
pip install joblib
pip install -e third_party/onnx
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug
pip uninstall numpy
pip install numpy~=1.22.2
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-7' -l='debug'
```

```txt
[2024-12-18 22:34:58,796] DEBUG: repo /home/sylvex/onnx-mlir/models reset   [2024-12-18 22:34:58,797] DEBUG: cmd=git reset --hard cwd=/home/sylvex/onnx-mlir/models
[2024-12-18 22:34:58,818] DEBUG: cmd=git clean -xdf cwd=/home/sylvex/onnx-mlir/models
[2024-12-18 22:34:58,858] DEBUG: cmd=find validated -type f -name *.tar.gz cwd=/home/sylvex/onnx-mlir/models
[2024-12-18 22:34:58,860] DEBUG: There are 184 models in the ONNX model zoo where 32 models are not checked because of old opsets or quantization.
[2024-12-18 22:34:58,861] DEBUG: Downloading https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz
[2024-12-18 22:34:58,861] DEBUG: cmd=curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mnist-7.tar.gz --output /home/sylvex/onnx-mlir/mnist-7.tar.gz cwd=/home/sylvex/onnx-mlir
[2024-12-18 22:35:00,202] DEBUG: Extracting the .tar.gz to /tmp/tmp1t9u5shu
[2024-12-18 22:35:00,203] DEBUG: cmd=find /tmp/tmp1t9u5shu -type f -name [^.]*.onnx cwd=None                                                            [2024-12-18 22:35:00,204] DEBUG: cmd=find /tmp/tmp1t9u5shu -type d -name test_data_set* cwd=None
[2024-12-18 22:35:00,205] DEBUG: Checking the model mnist-7 ...
[2024-12-18 22:35:00,205] DEBUG: cmd=/home/sylvex/onnx-mlir/utils/RunONNXModel.py --compile-args=-O0 --verify=ref --verify-every-value --load-ref=/tmp/tmp1t9u5shu/model/test_data_set_0 --model=/tmp/tmp1t9u5shu/model/model.onnx cwd=None
[2024-12-18 22:35:00,480] INFO: [mnist-7] check passed
[2024-12-18 22:35:00,480] DEBUG: [mnist-7] Temporary directory has been created at /tmp/tmpgnn8qyoc
Compiling the model ...
  took  0.15637316787615418  seconds.

Loading the compiled model ...
  took  0.00012977700680494308  seconds.

Reading inputs from /tmp/tmp1t9u5shu/model/test_data_set_0 ...
  - 1st input: [1x1x28x28xfloat32]
  done.

Running inference ...
  1st iteration, 0.002986923325806856, seconds
Reading reference outputs from /tmp/tmp1t9u5shu/model/test_data_set_0 ...
  - 1st output: [1x10xfloat32]
  done.

Verifying value of Plus214_Output_0:[1, 10] using atol=0.01, rtol=0.05 ...
  correct.

[Parallel(n_jobs=1)]: Done   1 tasks      | elapsed:    1.6s
[Parallel(n_jobs=1)]: Done   1 tasks      | elapsed:    1.6s
1 models tested: mnist-7

1 models passed: mnist-7

0 models failed:
```

# Universal Pybind

use python 3.10

```bash
cd build
python3.10 -m venv env
source env/bin/activate
cmake ..
make
cd ..
ln -s build/libpositWrapperPy.so
```

```bash
nm libpositWrapperPy.so | grep "getRawBit"

000000000006301f W _Z9getRawBitILm16ELm1EtET1_d
0000000000063509 W _Z9getRawBitILm32ELm2EjET1_d
00000000000639f7 W _Z9getRawBitILm64ELm3EmET1_d
0000000000062b31 W _Z9getRawBitILm8ELm0EhET1_d
```

```cpp
// libpositWrapperPy matches the name of the .so file
PYBIND11_MODULE(libpositWrapperPy, m) {
    bind_posit_functions<8, 0, uint8_t>(m, "8_0");
    bind_posit_functions<8, 1, uint8_t>(m, "8_1");
    bind_posit_functions<8, 2, uint8_t>(m, "8_2");
    bind_posit_functions<8, 3, uint8_t>(m, "8_3");
```

```bash
tree -I 'PositWrapper.egg-info|build|3rd_party'
.
├── CMakeLists.txt
├── include
│   └── positWrapperC.h
├── README.md
├── setup.py
├── setup.sh
├── src
│   ├── CMakeLists.txt
│   ├── positWrapperC.cpp
│   └── positWrapperPybind.cpp
└── test
    ├── CMakeLists.txt
    ├── test.cpp
    └── test.py
```

```python
import libpositWrapperPy as posit

print("posit(64, 0)")
rawBit = posit.getRawBit_64_0(1e+307)
print(bin(rawBit))
doubleVal = posit.getDouble_64_0(rawBit)
print(doubleVal)
```

```bash
posit(64, 0)
0b111111111111111111111111111111111111111111111111111111111111111
4.611686018427388e+18
posit(64, 1)
0b111111111111111111111111111111111111111111111111111111111111111
2.1267647932558654e+37
posit(64, 2)
0b111111111111111111111111111111111111111111111111111111111111111
4.523128485832664e+74
posit(64, 3)
0b111111111111111111111111111111111111111111111111111111111111111
2.0458691299350887e+149
```

# Future Works

- Make the test pipeline work
- Math lowering at `onnx-mlir` end
- Implement the math operation with different posit configuration
	- sitofp, exp, sqrt, tanh, erf
	- if we can't we can treat it double operation first