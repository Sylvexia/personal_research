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

# 2-27 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Initiate experiment design.
- tiny-yolov3, ssd failed
	- trying to use debugger to investigate, failed to load the MLIR file.

---

# What do we want?

- Currently ongoing the programming.
- Working directory that saves models and output.
	- Output serialization to file might need to cast posit to fp32
	- Serialization only support primitive type.
- Grabbing the dataset, apply preprocess and postprocess on input and output respectively.
	- Dataset might be big and specific to model, e.g. ImageNet
	- We might only able to run few tests since a single test require minutes to hours to run.

---

# What do we want?

- FP32 result as ground truth and compare them, Metrics we need:
	- Value Distribution: (latter works)
		- e.g. mobilenetv2-7 v.s. efficientnet-lite4-11
	- Classification: Accuracy, Top1, TopK.
	- Detection: mAP, IoU.
		- IoU 60%: 60% of BBox overlap with ground truth
		- mAP@`[0.5:0.95]`: mean AP of all classes with IoU threshold 0.5 - 0.95
		- AP: Area of precision-recall curve, higher the better.

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

posit(32, 2)
- mobilenetv2-7
  - elapsed: 26.6min
- efficientnet-lite4-11
  - elapsed: 65.35 mins

---

# Thnak you