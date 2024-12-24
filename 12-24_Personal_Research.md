
# TODO

- universal to third party
- Runtime
- Making a flow chart of end-to-end
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

- Separated universal wrapper
- between float and uint in posit in python version

# Runtime

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