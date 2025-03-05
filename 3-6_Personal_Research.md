
# Summary

- MobileNetv2 ImageNet dataset classification run through rough draft.
	- Highlight: found a way to sample from dataset and feed into model.

# ImageNet Classification Run Through

- Grab ONNX model automatically from the Internet.
- Randomly sample from dataset
- Image preprocess.
- Compile and Run FP/Posit model
- Export the FP/Posit output for later numerical analysis
# Layout

```bash
         9.3750000e-01, -1.5625000e-01, -1.5625000e-01,  1.2500000e+00,
         6.2500000e-01,  4.6875000e-02,  6.2500000e-02, -2.1875000e-01]])]
Ground Truth: 604
Predicted: 12
```
# Future works

- Experiment Result of Classification.
	- Numerical Analysis
# Mobile Net

curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz --output /home/sylvex/onnx-mlir/mobilenetv2-7.tar.gz
