Robin Hung

# Summary

- Initiate experiment design.
- tiny-yolov3, ssd failed
	- trying to use debugger to investigate, failed to load the MLIR file.

---
# What do we want?

- Working directory that saves models and output.
	- Output serialization to file might need to cast posit to fp32
	- Serialization only support primitive type.
- Grabbing the dataset, apply preprocess and postprocess on input and output respectively.
	- Dataset might be big and specific to model, e.g. ImageNet
	- We might only able to run few tests since a single test require minutes to hours to run.
- FP32 result as ground truth and compare them, Metrics we need:
	- Value Distribution: (latter works)
		- e.g. mobilenetv2-7 v.s. efficientnet-lite4-11
	- Classification: Accuracy, Precision, Recall, F1 Score
	- Detection: 

---
# Debugging

- There is a runtime program `run-onnx-lib` specific for GDB debugging
- Document states that it could log out the `mlir` file, however, I cannot get this to work.
- Other techniques such as rebuild pipeline, direct GDB, list conversion, still cannot find a way to solve this.

```bash
Debug/bin/onnx-mlir --preserveMLIR test_add.onnx
. ../utils/build-run-onnx-lib.sh
gdb Debug/bin/run-onnx-lib
(gdb) b run_main_graph
(gdb) run ./test_add.so
(gdb) list
1	builtin.module  {
2	  builtin.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) 
```

---

# Experiment:

mobilenetv2-7 (32, 2)
elapsed: 26.6min
(16, 2) failed

efficientnet-lite4-11 (32, 2)
65.35 mins
(16,2) 14.6 mins

`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --enable-posit --EmitONNXIR /home/sylvex/models/ssd-10/model.onnx -o meow`

`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --convert-krnl-to-affine --convert-arith-to-posit-func meow.onnx.mlir -o log`