
# Summary:

- Currently running numerical tolerance experiment
	- tinyyolo still running.
- Do we really need this experiment? Is it meaningful?

# Numerical tolerance

Scale the input by $2^0$ to $2^7$ (128), and measure the numerical error.

Posit(16,3) on MobileNetV2

```bash
=====Running iteration 0=====
MAE: 0.033319983672350645
=====Running iteration 1=====
MAE: 0.0320394943850115
=====Running iteration 2=====
MAE: 0.02313182172435336
=====Running iteration 3=====
MAE: 0.028150830119149758
=====Running iteration 4=====
MAE: 0.06595348196849227
=====Running iteration 5=====
MAE: 0.16082465364411475
=====Running iteration 6=====
MAE: 0.33457361839711663
=====Running iteration 7=====
MAE: 0.8125972071886063
```

---
# Numerical tolerance



---

# TinyYolo architecture

1. Multiply by 1/255
2. The following repeat 6 times
	1. Convolution
	2. Batch Normalization
	3. Leaky ReLu
	4. Max Pooling
3. Several combination of Convolution, Batch Normalization and Leaky Relu
4. Output

---
# TinyYolo: We have seen all those operators

1. Convolution: All tested model has it.
2. Batch Normalization: Except MNIST
3. Leaky Relu: All tested has ReLu, it just value < 0 needs to multiply by alpha.
4. Max Pooling: MNIST has it.

---
# Parameter

| mnist | mobilenetV2 | resnet-18  | tinyyolov2 |
| ----- | ----------- | ---------- | ---------- |
| 5998  | 3,539,138   | 11,699,112 | 15,867,889 |
