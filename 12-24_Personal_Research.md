
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