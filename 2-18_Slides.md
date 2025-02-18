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

# 2-18 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Bug fixes:
	- Fixed environment variable does not feed to the runtime.
	- Fixed multiple model runtime output error issue.
		- Important for object detection
- Model Usage Currently supported
	- Style Transfer
	- Super Resolution
	- Object Detection
	- Object Classification
	- Emotional Recognition

---

# New Model Result

- All result are posit (32, 2), all model are passed in  `atol=0.01`, `rtol=0.05`
- candy-8: 388.4min
	- Style Transfer
- super-resolution-10: 117.6min
	- Super Resolution
- version-RFB-640: 18.2min
	- Ultra Light Weight Face Detection.
- tinyyolov2-8: 246.3min
	- Object Detection

---

# Working Model Listing

- All result are posit (32, 2), all model are passed in  `atol=0.01`, `rtol=0.05`
- Style Transfer:
	- candy-8
- Super Resolution:
	- super-resolution-10
- Object Classification:
	- resnet-18

---

# Working Model Listing

- Object Detection:
	- version-RFB-640
	- tinyyolov2-8
- Emotional Recognition
	- emotion-ferplus-8: posit 32, 16 passed

---
# Future Works

- What experiment should we do?
	- What operators do a model have?
		- Numerical Analysis of supported operator.
			- We only have `+-*/` supported in posit
	- Lower bit and higher bit precision loss
		- Remember the ground truth precision is FP32
	- Model benchmark
		- Numerically fluctuation but argmax() works.
		- `[0.1, 0.2, 0.3]` v.s. `[0.2, 0.1, 0.3]`
- Should I spend some time on get gpt-2 running?