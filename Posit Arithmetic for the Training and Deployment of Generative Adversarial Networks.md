---
marp: true
theme: gaia
paginate: true
backgroundColor: #ffffff
color: #000000
style: |
  h1, h2, h3 {
    text-align: center;
  }
  section {
    font-size: auto;
  }
---

# Why GANs Are Hard to Train

---

## 1. Vanishing/Exploding Gradients  
- Generator & Discriminator gradients often become unstable.  
- Makes optimization difficult and slow.

---

## 2. Mode Collapse  
- Generator learns to produce limited patterns repeatedly.  
- Fails to generate diverse outputs.

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
