---
marp: true
theme: default
paginate: true
backgroundColor: white
color: black
style: |-
  h1, h2, h3 {
    text-align: center;
  }
  section {
    font-size: auto;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---

# Posit Arithmetic for the Training and Deployment of Generative Adversarial Networks

- Author: Nhut-Minh Ho, Duy-Thanh Nguyen, Himeshi De Silva, John L. Gustafson, Weng-Fai Wong, Ik Joon Chang,  National University of Singapore Kyung Hee University
- Professor: Peng-Shen, Chen
- Reporter: Yu-Chun, Hung

---
## Insight:

- This paper is trying to train and infer GAN model with lower bit data.
- Small numerical error can cause GAN training failure.
- Posit floating point data have more precision then normal floating data when value exponent close to zero.
- Scale the weight and loss to where posit good at to exploit the bit width.

---
## What is GAN?

- Consists of Generator and Discriminator Network:
	- Generator (G): Create fake data.
	- Discriminator (D): Distinguish real and fake data.
- Training:
	- Train G and D at the same time
		- G try to generate fake data such that D cannot distinguish.
		- D try to get better at judging real and fake data.

---
## What is GAN?

- Application:
	- Image generation
	- Data augmentation
	- Style Transfer

---
## What is Posit?

- Environment variables:
	- `n-bit`: length of bits, `es-val`: max length of exponent.
	- e.g. `posit<16, 3>`
- Format:
	- sign bit: 0 is +, 1 is -
	- regime bit: 
		- resizable bit, decide the $\text{useed} = 2^{2^{es}}$ exponent k
		- duplicate leading 0/1 and stop with opposite bit.
		- `110: k = 1`, `10: k = 0`, `01: k = -1`, `001: k = -2`
	- exponent: same as IEEE 754 exponent, but must be positive and no bias.
	- fraction bit: same as IEEE 754 fraction.
---
## What is Posit?
- Properties:
	- The carry of the posit:
		- 0,001,111,1 -> 
		- 0,01,000,00
		- regime bit shorter, fraction bit is longer.
	- If the full value exponent is closer to zero -> regime bit is shorter -> more space for fraction -> which mean more precision.
	- Under same `es-val`, conversion between n-bits requires only remove/pad zeros. 
![center h:240](paper_speech_image/posit_example.png)

---
## Contribution:

- The first to use non-standard 8-bit FP format to train GAN and 6-bit FP format for GAN inference.
- Fast approximation of `tanh(x)` function in posit.
- Software and Hardware Evaluation of GAN in posit and other FP format.

---
## Why GAN is hard to train? 

- No easy way to adopt small bit to train GAN.
	- Output of GAN is millions of pixels, which sensitive to numerical errors.
	- At that time, the only reliable method to train GAN was to use Nvidia mixed-precision framework to train GAN.
		- Nvidia O1 mode: Only use FP16 for GEMM operator, and others are FP32
	- No other proposal use bit width 8 to train GAN at that time.
		- Binary training, 8-bit training do exist but not include GAN.

---

## Why GAN is hard to train? 

- My insight
	- Sensitive to hyperparameter (e.g. learning rate)
	- Mode Collapse:
		- G Generate only generate same images.
	- Diminished gradient:
		- D is too good at judging, causing G gradient vanished and learn nothing.

---
## Numerical Properties of GAN training

- The height is frequency of $\log_2(|\text{values}|)$
- W: weight, A: Activation, G: Generator, D: Discriminator
- 0%, 50%, 100% means training epoch progress.
	- Value does not change much across epoch
- Weights are concentrated in $2^{-4}$ to $2^{-5}$, need to handle
- Activations are concentrated in $2^{-2}$ to $2^{0}$, no need to handle
![h:320 center](posit_gan_image/Value_Distribution.png)

---

## Proposed Method: System architecture

- Biased Encoder/Decoder:
	- For add/subtract "$t$" in exponent bit in posit data.
		- This scales power of 2.
		- Irrelevant to model architecture!
	- Encoder: $\{S, R, E + t, F\} \rightarrow \{P\}$ : scaler
	- Decoder: $\{P, t\} \rightarrow \{S, R, E - t, F\}$ : de-scaler
	- The weight is stored scaled format.
		- Before compute dot product, need to decode.
![h:240 center](posit_gan_image/system_arch.png)

---

## Proposed Method: System architecture

- Architecture:
  - W: weights, A: activation values, G: gradient, E: error
  - The dot product between `W * A` and `E * A` involves two `posit<8, 2>` multiplication and output is `posit<16,2>`
![h:300 center](posit_gan_image/system_arch.png)
---
## Proposed Method: System architecture

- Architecture (My insight)
	- `W * A -> A`
		- $a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$
			- The product of previous activation and weight plus bias is current activation value.
	- `E * A -> G`
		- $\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$
			- Gradient of loss is the product between error term and previous activation value.
![h:240 center](posit_gan_image/system_arch.png)
---

## Proposed Method: System architecture

- For operations value change is small like "other operations" and "weight updates", stored with `posit<16,2>` internally.
	- Standard low precision CNN training use bit width 16.
- The `es` value is kept 2, which can simply remove/add zero to convert between `posit<8, 2>` and `posit<16, 2>`
	- es value 1: fails to converge
	- es value 3: fraction accuracy is not enough
![h:320 center](posit_gan_image/system_arch.png)

---

## Proposed Method: System architecture

- The image is misleading in my opinion.
- What are the training steps?
  - Forward Pass:
    - Calculate activation value with weight, bias, previous activation value and activation value.
    - Calculate the loss with loss function metrics, which involves the difference between current and target output.
  - Backward Pass:
    - Gradient are computed for each weight, for steepest direction in loss function.
  - Weight Update:
    - The optimizer use gradient and learning rate to optimize the weight.
- Inference is just running the forward pass once.

---

## Proposed Method: Parameter Scaling

- In posit, value near exponent 0 has the most accuracy. Hence scale the value is helpful.
	- Weight scaling does not work in normal FP since the accuracy distribution is flat.
- The weight is decoded before multiply-add operation. Then encode after that.
	- Weights are kept scaled in weight update.
![center height:360](paper_speech_image/precision.png)
---

## Proposed Method: Parameter Scaling

- As mentioned before, t is for scaling factor in Encoder/Decoder
	- Encoder: $\{S, R, E + t, F\} \rightarrow \{P\}$
	- Decoder: $\{P, t\} \rightarrow \{S, R, E - t, F\}$
- How to decide scale t?
	- $\{ \text{bins}, \text{frequencies} \} = \text{histogram}\left( \log_2 \left( |W| \right) \right)$
	- $t = \left\lfloor 0 - \text{bins}\left[\arg\max(\text{frequencies})\right] \right\rfloor$, 
		- $t$ is integer, 
		- $\arg\max$ is the index of max array element.
	- Insight: $t$ shift the histogram such that the highest peak is 0
	![h:320 center](posit_gan_image/Value_Distribution.png)

---

## Proposed Method: Parameter Scaling

- From experiment: Across different GANs, the t value is `3~5`
	- Insight: es-bit is 2, so it freed 1 to 3 bits of fraction.
- Use only the first iteration histogram to set the `t`
	- Further calibration through iteration does not payoff.
	- The value distribution does not change much during training.

---
## Proposed Method: Loss Scaling

- Loss Scaling is standard approach in low precision training.
	- Prevents small gradient values from being rounded to zero.
- Steps:
	- **During** the gradient calculation, scale by s.
	- After the gradient is calculated, scale the value back.
	- Use the gradient to update weight.
	- Loss is a concept, since in back propagation, loss value itself is not used.
- Conventional method: (float)
	- Increase `s` until its overflow, then decrease - Nvidia Apex
- Proposed method: (posit)
	- Same as parameter scaling.

---
## Proposed Method: Loss Scaling
- X-axis: gradient value
- Y-axis: frequency
- Float loss scaling approach to overflow.

![center](posit_gan_image/loss_scale.png)

--- 
## Proposed Method: Fast Approx. of tanh(x)
- Most GANs use tanh as the output layer in the Generator
- Approximation: (formula)
- Correction: Set threshold and bias, and add up the quantity
![center](posit_gan_image/tanh_approx.png)
---
## Experiment

- FP32 accumulator for `mult-add` operation is enough for training.
	- FP64 is tried and does not help much, quire is not needed.
	- The output is then quantize to P16
- Compare: (specific spec needed)
	- Nvidia Apex O1: FP16
	- `QPytorch`: FP8
	- This Paper: P8

---
## Experiment

- Training quality:
	- The configuration is the same for different format.
		- random number generator state, default optimizer hyper parameters, and the number of epochs set by the original work
	- GAN train with different format would give different output.
		- Zebra stripe are different on the top row.
		- Same input vector in latent space give different output at the bottom row.
		- Need metrics to judge good or bad.
		![center h:240](posit_gan_image/GAN_Output.png)
---
## Experiment

- Training quality:
	- Metrics:
		- Inception Score: Not used?
		- `Frechet Inception Distance`: lower the better
			- Except ESRGAN: PSNR metrics
	- Result:
		- P8 outperform FP8
		- Scaling match fp16 training
		![center h:240](posit_gan_image/FID_Score.png)

---
## Experiment:

- Post-training deployment output quality
	- Use FP32 as baseline to compare.
	- P8 P6 use es = 1 for best quality
	- Metrics:
		- PSNR (Peak Signal-to-Noise Ratio): Higher the better
		- SSIM (Structural Similarity Index Measure): Higher the better

---
## Experiment:

- Post-training deployment output quality
	- Compare:
		- `+` means scaling
		- `T` means `tanh` approximation
	- Result:
		- `P6+T`, `P6+` have `SSIM` > `0.9`
			- High output quality
	![center h:320](posit_gan_image/infer_quality.png)

---
## Hardware Simulation

![](posit_gan_image/HardwareSimulation.png)

---
## Conclusion

- Presents ways to train GAN in 8-bit and deploy in 6-bit
- Using modified `PyTorch` allows to experiment with different training schemes.
- Hardware simulation shows posit has better energy and runtime.
- Once low precision accelerator emerged, the proposed method should be promising.

---
## Insight:

- Shift the scale of weight may help in low bit model training and inferencing.
	- Especially shift by power of 2 is easy for posit. (no bias like normal float)
- For same `es` value posit, conversion between them is easy.
	- Truncate fraction or append zero.
- For training, weight update is normally small. It must be handled
	- Learning rate: 2e-4