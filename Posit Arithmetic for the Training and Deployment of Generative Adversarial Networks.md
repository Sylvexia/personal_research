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

## Contribution:

- The first to use non-standard 8-bit FP format to train and 6-bit FP format to inference.
- Fast approximation of `tanh(x)` function in posit.
- Software and Hardware Evaluation of GAN in posit and other FP format.

---

## Why GAN is hard to train? 

No easy way to adopt small bit to train GAN.
- Output of GAN is millions of pixels which sensitive to numeric errors.
- At that time, the only method was to use Nvidia mixed-precision framework to train GAN.
	- Nvidia O1 mode: Only use FP16 for GEMM operator, and others are FP32
- No other proposal use bit width 8 to train GAN at that time.

---
## Numerical Properties of GAN training

- The height is frequency of log2(|values|)
- W -> weight, A -> Activation, G -> Generator, D -> Discriminator
- 0%, 50%, 100% means training epoch progress.
	- Value does not change much across epoch
- Weights are concentrated in 2^-4 to 2^-5, need to handle
- Activations are concentrated in 2^-2 to 2^0, no need to handle
![h:320 center](posit_gan_image/Value_Distribution.png)

---

## Proposed Method: System architecture

- Biased Encoder/Decoder:
	- For adding/subtracting exponent bit in posit data.
		- Irrelevant to model architecture!
	- t: Exponent bias.
	- Encoder: {S, R, E + t, F} -> {P}
	- Decoder: {P, t} -> {S, R, E - t, F}
	- The weight is stored scaled format. Decode then compute.
![h:320 center](posit_gan_image/system_arch.png)

---

## Proposed Method: System architecture

- Architecture:
  - W: weights
  - A: activation values
  - G: gradient
  - E: error
- dot product between `W * A` and `E * A`
	- Involves two `posit<8, 2>` multiplication and output is `posit<16,2>`
![h:320 center](posit_gan_image/system_arch.png)

---

## Proposed Method: System architecture

- For "Other operations" and "weight updates", value change is small. Stored with `posit<16,2>` internally.
	- Standard low precision CNN training use bit width 16.
- The `es` value is kept 2, which can simply truncate or concatenate zero to convert between `posit<8, 2>` and `posit<16, 2>`
	- es value 1: fails to converge
	- es value 3: fraction accuracy is not enough
![h:320 center](posit_gan_image/system_arch.png)

---

## Proposed Method: System architecture

- What is the training steps?
  - Forward Pass:
    - From input to last layer to give output.
  - Backward Pass:
    - Calculate the error from predict output and target.
    - Gradient are computed for each weight, for steepest direction in loss function.
  - Weight Update:
    - The optimizer use gradient and learning rate to optimize the weight.
- Inference is just running the forward pass once.

---

## Proposed Method: Parameter Scaling

- In posit, value near exponent 0 has the most accuracy. Hence shift the value is helpful.
	- Weight scaling does not work in normal FP since the accuracy distribution is flat.
- The weight is decoded before multiply add operation. Then encode after that.
	- Weights are kept scaled in weight update.
![[Pasted image 20241021234214.png]]
---

## Proposed Method: Parameter Scaling

- As mentioned before, t is for scaling factor in Encoder/Decoder
	- Formula:
- How to decide integer t?
	- Formula:

---

## Proposed Method: Parameter Scaling

- Across different GANs, the t value is `3~5`
- Use only the first iteration histogram to set the `t`
	- Further calibration through iteration does not payoff.
	- Maybe it's because the value distribution does not change much during training?

---
## Proposed Method: Loss Scaling

- Standard approach in low precision training.
	- Prevents small gradient values from being rounded to zero.
- Scale the loss by `s`, gradient would also be scaled.
- Gradient must be unscaled before weight update.
- Conventional method: (float)
	- Increase `s` until its overflow, then decrease - Nvidia Apex
- Proposed method: (posit)
	- Shift the center of the distribution towards the range of posits that have the highest accuracy.

---
## Proposed Method: Loss Scaling

![[Pasted image 20241022085627.png]]

--- 
## Proposed Method: Fast Approx. of tanh(x)

- Most GANs use tanh as the output layer in the Generator
- Approximation: (formula)
- Correction: Set threshold and bias, and add up the quantity
![[Pasted image 20241022091705.png]]
---
## Experiment

- FP32 accumulator for `mult-add` operation is enough for training.
	- FP64 is tried and does not help much, quire is not needed.
	- The output is then quantize to P16
- Compare: (specific spec needed)
	- Nvidia Apex O1: FP16
	- QPytorch: FP8
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
		- (fig 8.)
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
		- (table 1)

---
## Experiment:

- Post-training deployment output quality
	- Use FP32 as baseline to compare.
	- P8 P6 use es = 1 for best quality
	- Metrics:
		- PSNR (Peak Signal-to-Noise Ratio): Higher the better
		- SSIM (Structural Similarity Index Measure): Higher the better
	- Compare:
		- `+` means scaling
		- `T` means `tanh` approximation
		(fig 9 a b)
	- Result:
		- P6+T, P6+ have SSIM > 0.9
			- High output quality

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
- 