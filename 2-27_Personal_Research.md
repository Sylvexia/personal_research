Robin Hung

# Summary

- Initiate experiment design on Classification and Object Detection
- tiny-yolov3 failed, trying to use debugger to investigate, failed to load the mlir file.
# What do we want?

- Grabbing the dataset, apply preprocess and postprocess on input and output respectively.
- Run
- Working directory that saves models and output.
	- output reference serialization to file.
- FP32 result as ground truth and compare them.
	- metrics for 

# Experiment:

mobilenetv2-7 (32, 2)
elapsed: 26.6min

(16, 2) failed