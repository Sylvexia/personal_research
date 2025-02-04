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

# 1-21 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Lowering 20+ operations in wrapper and mlir.
- Still investigating what model can be run... 
- (Sorry, recently been pretty busy for private stuff...)
---

# Language Models

- Models:
	- BERT-Squad: This model answers questions based on the context of the given input paragraph.
	- BiDAF: This model is a neural network for answering a query about a given context paragraph.
		- Smallest but cannot run even in fp32 case
	- GPT-2: Transformer-based language model for text generation.
	- GPT-2 with Beam Search Generation: Transformer-based language model for text generation.
	- RoBERTa: Transformer-based language model for text generation.
  
---

# Language Models

- Models:
	- T5: provide great flexibility and provide better semantic understanding through the training of multiple tasks at once.
- Summary: It's unlikely to be able to run any language models, since other models is quite large.

---
# Visions

- Category:
	- body analysis: age/gender classification, face recognition, emotion recognition
		- face detection: RFB-320
			- The result can output from model, but the conversion from raw posit bit failed.
	- classification: output the label of the image
		- we can run resnet-18 (last week report)
	- object detection: get the bounding box and label of objects in image. 
		- yolov4
			- Lowering issue, currently investigating...
---
# Visions
- Category:
	- style transfer: mix 2 images together.
	- super resolution: upscale image.
    	- After running whole night, the models crash the terminal session.
---
# Supported operation

- 22+ arith and math lowering on both wrapper and mlir side
	- Not fully tested on both side since there are a lot of operations
- Supported listing: 
	- arith: add, sub, mul, div, cmp, select, fptosi, sitofp, maxnum, minnum, max, min, neg
	- math: abs, sqrt, rsqrt, exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, erf, log, floor, ceil, trunc, round
---
# Future Works

- Validate lowered operations and models output.
- Design experiment that should be put into the master thesis.