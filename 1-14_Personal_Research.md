洪祐鈞

# TODO

- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- Get this work
	`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v`
- ` %516 = "onnx.Reshape"(%512#0, %515) {allowzero = 0 : si64, onnx_node_name = "Reshape_78"} : (tensor<?x?x768xf32>, tensor<4xi64>) -> tensor<?x?x12x64xf32>`
# Summary

- Now the config can run at project folder

gmp-6.2.1