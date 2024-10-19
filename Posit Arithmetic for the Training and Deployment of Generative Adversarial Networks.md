---
marp: true
theme: gaia
paginate: true
backgroundColor: 
color: 
style: |-
  h1, h2, h3 {
    text-align: center;
  }
  section {
    font-size: auto;
  }
---

# Posit Arithmetic for the Training and Deployment of Generative Adversarial Networks

---

## What is GAN?

- Consists of Generator and Discriminator Network:
	- Generator (G): Create fake data.
	- Discriminator (D): Distinguish real and fake data.
- Training:
	- Train G and D at the same time
		- G try to generate fake data such that D cannot distinguish
		- D try to get better at judging real and fake data.
- Application:
	- Image generation
	- Data augmentation
	- Style Transfer

---

## Contribution:

- The first to use non-standard 8-bit FP format to train and 6-bit FP format to inference.
- Fast approximation of tanh(x) function in posit.
- Software and Hardware Evaluation of GAN in posit and other FP format.

---

## Why GAN is hard to train? 

No easy way to adopt small bit to train GAN.
- Output of GAN is millions of pixels which sensitive to numeric errors.
- At that time, the only method was to use Nvidia mixed-precision framework to train GAN.
	- Nvidia O1 mode: Only use FP16 for GEMM operator

---

## 3. Non-Convergence  
- Both models compete without reaching equilibrium.  
- Leads to oscillations in performance.

---

## 4. Hyperparameter Sensitivity  
- GANs are highly sensitive to learning rates & batch sizes.  
- Small changes can destabilize training.

---

## 5. Evaluation Challenges  
- No perfect metric for assessing GAN output quality.  
- Requires manual inspection and multiple metrics (e.g., FID, IS).

---
