2018-fall
Zishen Wan, Eric Mibuari, En-Yu Yang, Thierry Tambe
Harvard University
Cambridge, MA

[[speech_recognition.pdf]]

## Intuition

- Posit: A new ways to represent floating-numbers, good at represent the |number| close to 1
- Text-to-Speech model is RNN, attention based model.
- Inference: Given input to existing model and get the output.
- Quantize: Fit big data to small data.
## Why do I choose this paper?

- My research focus is on integrating posit floating point arithmetic in to MLIR infrastructure, specifically on deep learning application.
	- MLIR: Framework of Compiler Framework
- Another research topic of lab is optimizing or quantize the Whisper model, which a lot of the operator e.g. Attention layer is similar to this.
## How does Posit Works?

- Format:
	- ![[posit_format.png]]
- Parameter: 
	- Posit<total_bits, es_bits>
	- Example: Posit<32,3>
		- Total of 32-bit. 32-bit posit
		- es: 3
			- At most 3 bit of exponent
			- 3-bit of exponent as a pack
- Formula:
	![[formula.png]]
	- 00000000: Zero
	- 10000000: NaR (Not a Real)
	- useed = 2^(2^es)
- Bit Representation
	- sign term: 
		- 0 -> +
		- 1 -> -
	- exponent term: 
		- 101 -> e = 5
		- 2^e , no Bias like IEEE754
		- at most es-bit
	- fraction term: 
		- 1011 -> 1 + (0.5 + 0.125 + 0.0625)
		- f = 1.f1f2f3f4... like IEEE754, 
		- but no subnormal number.
	- regime term: 
		- Find until the first opposite bit
		- 001 -> k = -2, 
		- 01 -> k = -1 
		- 10 -> 0, 
		- 110 -> k = 1, 
		- 1110 -> k = 2, 
		- k value intuitively treat as carry of pack of exponent
		- Decide useed^k = 2^(2^es)^k
		- __builtin_clz() -> x86 instruction LZCNT

## Insight on posit

"0,0001,111,11111111" next posit is:
"0,00001,000,0000000"

- We need larger precision at smaller exponent.
	- We have more fraction bit at smaller exponent.

## Example

- Given posit<16,3> (16-bit, es = 3)
![[posit_example.png]]
- sign term: 0 -> positive -> +
- regime term: 0001 -> k = -3, -> 256^-3
	- useed = 2^(2^es) = 256
- exponent term: 101 = 5 -> 2^-5
- fraction term: 1+ (0.5 + 0.25 + 0.0625 + ...)

### Convert from posit to real number:

- Add check (zero, NaR..) exception
- Check regime bit length -> exp -> fraction
	- exp and fraction bit may not exist.
- Multiply those sign, regime, exponent, fraction terms

### Convert from real number to posit

Example:

convert 694.20 to posit<32, 2>
es = 2
useed = 2^(2^2) = 2^4

694.20 = 2^9 \* 1.359

9/4 = 2 ... 1 -> k = 2, e = 1
- regime bit = 1110
- exponent bit = 01

positive -> sign bit = 0
1.359 = 1 + (2^-2 + 2^-4 + 2^-5 + 2^-7 + 2 ^-8 ...)
- fraction bit = 0101101100011001100110011

In paper: 3 cases:
- |value| > useed
- useed > |value| > 1
- 1 < |value| < 0

original paper non-sense:
![[wrong.png]]
## Larger Dynamic

- Posit have larger dynamic range
![[Dynamic_Range.png]]

# Accuracy

- IEEE FP32: 23-bit fraction + 1(1.xxx fraction)
- Posit 32: 32 - 1(sign) - 2(regime) - 2(exponent) + 1(1.xxx fraction)
- IEEE FP16: 10-bit fraction + (1.xxx fraction)
- Posit 16: 16 - 1(sign) - 2(regime) - 1(exponent) + 1(1.xxx fraction)
![[DecimalAccuracy.png]]

# Posit(32, 2) accuracy v.s. IEEE float 32 

- Posit has more accuracy when the value's 2's exponent is close to 0
![[precision.png]]
(ref: https://spectrum.ieee.org/floating-point-numbers-posits-processor)

## Posit Precision

- Observation: 
	- shorter regime bits allows more fraction bits
	- Value distribution is close together exponentially around 0. 
![[posit_construction.png]]
## Abstract

- RNN model is huge for memory aspect.
- Quantize the mode is good for deploy to edge device.
- Research various kind of floating point format on speech-to-text model inference performance
- Result shows posit<8,0> is the best for aggressive quantization.
- Posit hardware is more efficient then fixed-point based, in terms of area and power cost.

## Contribution

- Develop a Python-based framework for converting between Float, fixed-point and posit numbers.
- Compare and evaluate the posit and various data types in neural speech recognition inference.
- Design hardware unit of posit, fixed point and floating point.
- Overall hardware prototype of speed-to-text inference.

- Sad, it seems like it's not open-source

## Overview of common numerical data type

![[fp_format.png]]

- BF16: Used in Google cloud TPU and Intel AI processors
- Fixed-point: 0.1 + 0.2 != 0.30000000000000004
	![[meme_decimal.png]]
- low precision fixed-point arithmetic become DL quantization in inference.

## Other FP Format

![[FP8_News.png]]
![[FP8_Paper.png]]
![[FP4_news.png]]
## Experiment

- Train 2 speech recognition model
	- Framework: OpenNMT toolkit
	- Dataset:  LibriSpeech corpus: 
		- 1,000 hours of audiobooks
	-  Model architecture follows DeepSpeech3 specifications
	- Substitute batch normalization to layer normalization.
		![[Normalization.png]]
		ref:(https://medium.com/@zljdanceholic/groupnorm-then-batchnorm-instancenorm-layernorm-e2b2a1d350a0)
	- Networks were not retrained after quantization was applied
	- There's no specific words describe how the quantization is done
		- From the context, the conversion is based on real value.
		
- Model 1:
	- MLP attention model
	- Encoder: 
		- 4 layer of bidirectional-GRU (Gated Recurrent Unit)
		- ![[GRU.png]]
			- Ref(https://vtiya.medium.com/gru-vs-bi-gru-which-one-is-going-to-win-58a45ede5fba)
			- 1024 hidden units
		- 2 downsampling pooling layer
	- Decoder: 1 forward-only decoder with 512 hidden unit
	- 20M parameter

- Model 2:
	- General attention model
	- Encoder:
		- 5 layer of uni-directional LSTM
			- 800 hidden units
		- 4 down-sampling pooling layer
	- Decoder: 2-layer forward-only decoder with 512 hidden unit
	- 30M parameter

![[accuracy_loss.png]]

WER(Word Error Rate)
- The lower, the better

Observation:
- posit<8,0> have small accuracy loss compare to original IEEE FP32
	- Original paper of posit actually mentioned posit
- For 8-bit data type, posit<8,0> outperform others.
- For all 16-bit data-type, the accuracy is about the same.
	- Expected since
		- weight is [-2.5, 2.5] for model 1
		- weight is [-2, 2] for model 2
-  8-bit fixed-point<2,6> is not enough for number more than 2 but better than fixed-point<3,5>
	- Denser distribution in [-2, 2] in model 1
- After 1 epoch of re-train and re-quantize
	- model 1 accuracy: (19.10 -> 18.84) vs 18.80

## Hardware Aspect

- Create hardware for:
	- Conversion between posit, floating and fixed point number
	- MAC (**M**ultiply **Ac**cumulate)
		- a <- a+b\*c
		- Good for convolution, matrix operation.

### Hardware Spec:

![[Conversion_spec.png]]
![[mac_spec.png]]
- 8-bit posit adder is approximately 25% of 32-bit float-point adder
- Posit adder is smaller than fixed-point adder

- 90nm library. 100MHz
- My 64-bit double multiplier: 5GHz, 111.2mW, 6.324um^2, 60 cycle. 16nm TSMC (No handling denormal)
- Original paper does not state that it’s single cycle.

![[No_Denormal.png]]
- ref:https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/
- +20% speedup
### Question:

1. How many cycle of this hardware? What standard of the implementation did they follow?
	
	![[rethink_hardware.png]]
	ref: [Rethinking floating point for deep learning](https://arxiv.org/abs/1811.01721)

## Prototype LSTM ENGINE

### LSTM Formula

![[LSTM_Formula.png]]
- x: input
- h: hidden state
- c: cell state
- First 4 formula is similar to fully-connected layers, just matrix multiplication and addition. For pre-calculations.
- Activation function is tanh and sigmoid, perform after summation of matrix multiplication
- Fifth formula is cell state, depend on previous state
- Last formula is for the next hidden state

### Accelerator Architecture

![[LSTM_ACCEL_ARCH.png]]
- Datapath:
	- Use cache buffers for input, hidden state, and weight. use 8MACs to calculate the matrix-multiplication and 4 accumulator for addition.
	- Feed 4 result to calculate activation.


- Activation function tanh and sigma is done by piecewise approximation.
	- Using straight line segment to approximate the function. 
	- ![[Pieceswise_sigmoid.png]]

### Summary of LSTM accelerator 

- Synthesized the accelerator from ststemC code to RTL in verilog and verify the result.
-  Bottleneck: 
	- Storage of weight matrices.
	- taking advantage of weight reuse across timesteps.

## Conclusion:

- posit<8, 0> is suit for speech recognition inference.
- posit based hardware would have less area and power usage.
- posits present as appealing solution for compress DNN on edge and warrant further studies.

## Question

- Does it work on other type of model?
	- ref: [Rethinking floating point for deep learning](https://arxiv.org/abs/1811.01721)
		- batch normalization fused into affine layers.
		- float32 parameters and network input are converted to our formats via round-to-nearest-even
		![[Pasted image 20240826232322.png]]
	- ref: Deep PeNSieve: a Deep learning framework based on the Posit Number System
		- Linear mapping quantization
		- ![[Pasted image 20240826235843.png]]
	- Analyze different weight distribution on different operator and different model?
		- https://www.researchgate.net/publication/329798596_Fast_Adjustable_Threshold_For_Uniform_Neural_Network_Quantization
			![[weight_distribution.png]]
			- Distribution of weights of ResNet-50 neural network before the quantization procedure (on the left) and after it (on the right).
- Different quantize ways, like how it maps to uint8.
	- Dynamic/Static ways. (w/ and w/o callibration)
- Different hardware implementation can make the area and power have different outcome.

## Thoughts

- You can make your whole model binary as long as your network has enough number of nodes:
	- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
	- ![[Pasted image 20240827031644.png]]
- You can 1-bit LLM, if you do quantization every linear layer and good at math:
	- ![[Pasted image 20240826233140.png]]