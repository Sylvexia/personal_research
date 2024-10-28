# Future Path

1. LLVM APFloat posit interface
2. MLIR Type posit Attribute
3. ONNX Type post interface (and quantize operator?)
4. onnx model posit converter (f32 -> posit8)
5. onnx-mlir posit dialect and arith-dialect lowering
6. accuracy verifier.

## What format should we implement?

Original posit paper states that posit<8,0> is good for deep learning application.

- posit<bit, es>

![[note_image/orig_posit_nn_training.png]]

in "Study of Posit Numeric in Speech Recognition Neural Inference"
We have good result:
![[paper_speech_image/accuracy_loss.png]]

but in "Rethinking floating point for deep learning"
The dynamic range of posit<8,0> is not enough:

![[note_image/Pasted image 20240827095150.png]]

Distribution of model parameters:
![[note_image/weight_distribution.png]]

If we were to map the value, how do we do this?

## How do we implement to LLVM/MLIR

In order to have posit type attribute, we need LLVM APFloat interface, and MLIR Builder interface

Current in LLVM APFloat implement way:

![[note_image/APFloat_impl.png]]

- Currently if we were to add other es value posit, e.g. posit<8, 2>, we cannot simply use parameter and template object to generate the type. (it would be hard)
	- We probably only be able to do:
		```cpp
		case Posit8Es0:
			return a;
		case Posit8Es1:
			return b;
		```

Current status: Compile completed. Yet to verify.

Now there's onnx-mlir part yet to implement.

## Quantization?

fp32 -> posit8 (round to nearest even)

What do we mean by quantization?

- `val_fp32 = scale * (val_quantized - zero_point)`
    
- asymmetric quantization:
    
    - `scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)`
- symmetric quantization:
    
    - `scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)`
- Dynamic quantization: Quantization with calibration
- Static quantization: Quantization with calibration

Two ways to represent quantized ONNX model:

- Operator-oriented (QOperator):
	- All the quantized operators have their own ONNX definitions, like QLinearConv, MatMulInteger and etc.
	- Dynamic quantization calculates the quantization parameters (scale and zero point) for activations dynamically. These calculations increase the cost of inference, while usually achieve higher accuracy comparing to static ones.
- Tensor-oriented (QDQ(Quantize and DeQuantize)):
	- Static quantization method first runs the model using a set of inputs called calibration data.

Scope of quantize
- Per tensor
	- all the values within the tensor are quantized the same way with the same quantization parameters
- Per channel
	- for each dimension, typically the channel dimension of a tensor, the values in the tensor are quantized with different quantization parameters.

## Implement the posit pass:

Current we have to famous posit implementation:
- SoftPosit
- Universal

Actually, the data hold by both library is not the same. posit<32,2>(-31):
- SoftPosit: 10011100010000000000000000000000
- Universal: 11100000010000000000000000000000

The implementation detail would be investigated further.

- arith mapping
	- implementing the pass like this would be like porting the library in assembly.
- function mapping
	- If I can get it working it should be fast, but this is not a good way of doing it.

## How to add support?

- Most of the ops only support for f16, f32, f64, not even bf16
- Does it implicit convert the type is a question.

PositONNXOp:

round
Conv->QConv

ONNX dialect -> pass -> krnl (loop representation)
![[note_image/Pasted image 20240827130018.png]]