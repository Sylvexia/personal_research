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

# 3-13 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Bug Fix: Input does not load properly.
- Feature added: 
	- JSON file storage
	- Graph
- Experiment on `mobilenetv2`, sample = 10, es = 0, 1, 2
	- posit8 and posit(16, 0) failed
	- posit32 numerical result is almost the same as fp32
	- posit16 other than es=0, has acceptable output
---

# Average mean average error

![center h:480](note_image/mobilenet-imagenet1000_mae.png)

---

# Average root mean square error 

![center h:480](note_image/mobilenet-imagenet1000_rmse.png)

---

# top1 accuracy 

![center h:480](note_image/mobilenet-imagenet1000_top1_accuracy.png)

---

# top5 accuracy

![center h:480](note_image/mobilenet-imagenet1000_top5_accuracy.png)

---

# Future Works

- MNIST model
	- Operation is more simple may show more of posit.
- Object Detection model
  - Some model cannot compile

---

# Happy Birthday
