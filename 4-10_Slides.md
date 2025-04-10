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

# 4-10 Personal Research
## Presenter: Yu-Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Fixed the Posit NaR issue, at least for MNIST model.
	- Fixed by replace the hardcoded posit constant conversion in pass.

---

# Bug fix approach

- Add the CCU posit library as 3rd party dependency.
- Link the posit library with CMake. (Which is excruciating to integrate)
- Use the `getRawBit()` template function that was used for python wrapper.
- Integrate with `APFloat::convertToDouble()`.

---

# MNIST Average Top-1-Accuracy (N = 250)

![center h:480](note_image/mnist_top1_accuracy_v2.png)

---

# MNIST Average Top-5-Accuracy (N = 250)

![center h:480](note_image/mnist_top5_accuracy_v2.png)

---

# MNIST Average MAE (N = 250)


![center h:480](note_image/mnist_mae_v2.png)

---

# MNIST Average RMSE (N = 250)

![center h:480](note_image/mnist_rmse_v2.png)

---

# Future Works

- Finish object detection experiment.
- Full Statistics for classification experiments so far.
- Next week lab meeting group report

---

# Thank you!