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

# 3-6 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- MobileNetv2 ImageNet dataset classification run through rough draft.
	- Highlight: found a way to sample from dataset and feed into model.

---

# ImageNet Classification Run Through

- Grab ONNX model automatically from the Internet.
- Randomly sample from dataset
- Image preprocess.
- Compile and Run FP/Posit model
- Save the FP/Posit output for later numerical analysis

---

# Layout

```bash
Ground Truth: 604
Predicted: 12
```

```bash
MAE: 2.735914750029333
RMSE: 3.490900085600701
Ground Truth Top 5 label indices: [883 572 604 969 503]
Posit8_2 Top 5 label indices: [456  75  12  69 921]
```
- What's the meaning of precision, how to analyze.

---

# Future works

- Experiment result
- Various Model

---

# Thank you