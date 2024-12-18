
# TODO

- Posit conversion
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
export ONNX_MLIR_HOME=/home/sylvex/onnx-mlir/build/Debug/****
pip uninstall numpy
pip install numpy~=1.22.2
python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='gpt2-10' -l='debug'
```