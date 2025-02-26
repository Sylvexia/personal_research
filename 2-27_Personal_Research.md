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
	- 
	- Classification: Accuracy, Precision, Recall, F1 Score
	- Detection: 

# Debugging

- There is a runtime specific for GDB debugging

```bash
Debug/bin/onnx-mlir --preserveMLIR test_add.onnx
. ../utils/build-run-onnx-lib.sh
gdb Debug/bin/run-onnx-lib
(gdb) b run_main_graph
```

# Experiment:

mobilenetv2-7 (32, 2)
elapsed: 26.6min

(16, 2) failed

efficientnet-lite4-11 (32, 2)
65.35 mins