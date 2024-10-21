What do we mean by quantization?

- `val_fp32 = scale * (val_quantized - zero_point)`
- asymmetric quantization:
	- `scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)`
- symmetric quantization:
	-  `scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)`

- Dynamic quantization: Quantization with calibration
- Static quantization: Quantization with calibration

Not all the operator can be quantized: 

Pytorch Example:
![[pytorch_quantize.png]]

[ONNX quantize article](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

Two ways to represent quantized ONNX model:

- Operator-oriented (QOperator):
	- All the quantized operators have their own ONNX definitions, like QLinearConv, MatMulInteger and etc.
	- Dynamic quantization calculates the quantization parameters (scale and zero point) for activations dynamically. These calculations increase the cost of inference, while usually achieve higher accuracy comparing to static ones.
- Tensor-oriented (QDQ(Quantize and DeQuantize)):
	- Static quantization method first runs the model using a set of inputs called calibration data.

- Per tensor
	- all the values within the tensor are quantized the same way with the same quantization parameters
- Per channel
	- for each dimension, typically the channel dimension of a tensor, the values in the tensor are quantized with different quantization parameters.

- Post Training Quantization (PTQ)
	- Quantize after training.
- Quantization Aware Training (QAT)
	- Training with additional epoch while quantizing.