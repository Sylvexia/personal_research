洪祐鈞

# TODO

- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- Get this work
	`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v`
- `loc("onnx.Reshape"("Reshape_78")): error: failed to materialize conversion for result #0 of operation 'memref.reinterpret_cast' that remained live after conversion`
- ` %516 = "onnx.Reshape"(%512#0, %515) {allowzero = 0 : si64, onnx_node_name = "Reshape_78"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>`
- ` sylvex@sylvex-Aspire-A715-51G:~/onnx-mlir/build$ cat onnxDump.onnx.mlir | rg "%516"%516 = "onnx.Reshape"(%512#0, %515) {allowzero = 0 : si64, onnx_node_name = "Reshape_78"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32> %517 = "onnx.Transpose"(%516) {onnx_node_name = "Transpose_79", perm = [0, 2, 1, 3]} : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>`
# Summary

- Now the config can run at project folder
- gmp-6.2.1