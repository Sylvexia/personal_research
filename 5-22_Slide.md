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

# 5-22 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Try to delete the ONNX model node to get each node output to debug tinyyolov2
  - Half way through?
- Possibility of fp16 accuracy.

---
# The current way:

1. Load the ONNX model
2. Build a consumer map for traversal and locate the target deletion node
3. Find all downstream nodes (BFS) and remove them
4. Append last remaining node’s output as graph output
  1. The shape is odd, thinking a way how to get the dimension.
5. Save the ONNX Model

---
# Not deleted

![center h:480](note_image/Pasted%20image%2020250522053448.png)

---
# Deleted

![center h:480](note_image/Pasted%20image%2020250522053301.png)

---

# ONNX FP16

```python
model_fp16 = float16.convert_float_to_float16(model_fp32)
session = ort.InferenceSession("model_fp16.onnx", providers=['CPUExecutionProvider'])
inputs = {session.get_inputs()[0].name: np.random.randn(1, 1, 28, 28).astype(np.float16)}
```

```bash
python runfp16.py
Output: [[-4.48, 3.373, 0.3242, 2.152, 0.2324, 1.377, -2.598, 3.748, 0.8613, -4.355]]
```

---
# Future Works

- FP16 results.
- Master Thesis.