
# Summary

- MobileNetv2 ImageNet dataset classification run through rough draft.
	- Highlight: found a way to sample from dataset and feed into model.

# ImageNet Classification Run Through

- Grab ONNX model automatically from the Internet.
- Randomly sample from dataset
- Image preprocess.
- Compile and Run FP/Posit model
- Export the result for later 
# Future works

- Experiment Result of Classification.
- 
# Mobile Net

curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz --output /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz
