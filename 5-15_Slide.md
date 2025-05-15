---
marp: true
theme: default
paginate: true
header: 
footer: 
style: "h1, h2, h3 {\r  text-align: center;\r}

  pre, code {\r  background-color: #ffffff;\r    \r  color: #2d2d2d; \r  \r  font-size: auto;\r }\r

  section {\r  font-size: auto;\r}\r

  img[alt~=\"center\"]\ 

  {\r  display: block;\r  margin: 0 auto;\r}"

---

# 5-15 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Currently still investigating how to delete nodes in ONNX, to debug the tinyyolov2 model.
	- By deleting the node from back and run the inference, we can list out the numerical error from every node.
- The followings experiment is ResNet models with different depth
	- ResNet 18, 34, 50, 101, 152, the number means the number of layer.
	- The number of sample is 10. (Or we'll not getting the result by today)
	- For the current data, we cannot find significant result as the layer increase.
		- Model architecture matters more?

---

# 8-bit Mean Average Error

| MAE\Model  | (8, 0) | (8, 1) | (8, 2) | (8, 3) |
| ---------- | ------ | ------ | ------ | ------ |
| ResNet-18  | 2.12   | 1.93   | 2.44   | 2.51   |
| ResNet-34  | 1.94   | 1.67   | 2.06   | 2.13   |
| ResNet-50  | 1.89   | 1.90   | 1.90   | 1.92   |
| ResNet-101 | 1.83   | 1.85   | 1.83   | 1.83   |
| ResNet-152 | 1.82   | 2.20   | 2.04   | 1.90   |

---
# 16-bit Mean Average Error

| MAE\Model  | (16, 0) | (16, 1) | (16, 2) | (16, 3) |
| ---------- | ------- | ------- | ------- | ------- |
| ResNet-18  | 2.60e-2 | 5.57e-3 | 7.41e-3 | 1.42e-2 |
| ResNet-34  | 2.33e-2 | 5.42e-3 | 7.11e-3 | 1.33e-2 |
| ResNet-50  | 1.25e-2 | 5.29e-3 | 7.16e-3 | 1.49e-2 |
| ResNet-101 | 1.58e-2 | 5.19e-3 | 7.50e-3 | 1.52e-2 |
| ResNet-152 | 8.99e-2 | 4.97e-3 | 7.19e-3 | 1.47e-2 |

---

# 8-bit Accuracy

| Acc.\Model | (8, 0) | (8, 1) | (8, 2) | (8, 3) |
| ---------- | ------ | ------ | ------ | ------ |
| ResNet-18  | 0      | 0      | 0      | 0      |
| ResNet-34  | 0      | 20%    | 0      | 0      |
| ResNet-50  | 0      | 0      | 20%    | 0      |
| ResNet-101 | 0      | 0      | 0      | 0      |
| ResNet-152 | 0      | 0      | 0      | 0      |

---
# 16-bit Accuracy

| Acc.\Model | (16, 0) | (16, 1) | (16, 2) | (16, 3) |
| ---------- | ------- | ------- | ------- | ------- |
| ResNet-18  | 100%    | 100%    | 100%    | 100%    |
| ResNet-34  | 100%    | 100%    | 100%    | 100%    |
| ResNet-50  | 100%    | 100%    | 100%    | 100%    |
| ResNet-101 | 100%    | 90%     | 90%     | 100%    |
| ResNet-152 | 100%    | 100%    | 100%    | 90%     |

---
# Future Works

- ONNX node removal experiment.
- Master Thesis abstraction.