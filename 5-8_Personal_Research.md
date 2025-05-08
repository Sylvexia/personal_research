
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
