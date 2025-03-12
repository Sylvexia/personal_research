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
# Experiment Result


---

# Future Works

- MNIST model
	- Operation is more simple may show more of posit.

python run.py --n-bit=16 --es=0 --n-sample=10 > log16_0 2>&1

python numerical.py --n-bit=16 --es=0 --n-sample=10 > log16_0_result 2>&1

n bit 8 16 32
es 0 1 2

Average MAE: nan
Average RMSE: nan
Average Top-1 Accuracy: 0.0
Average Top-5 Accuracy: 0.02

Average MAE: nan
Average RMSE: nan
Average Top-1 Accuracy: 0.0
Average Top-5 Accuracy: 0.02

Average MAE: 2.613018201362924
Average RMSE: 3.4328833767783697
Average Top-1 Accuracy: 0.0
Average Top-5 Accuracy: 0.0

Average MAE: nan
Average RMSE: nan
Average Top-1 Accuracy: 0.0
Average Top-5 Accuracy: 0.0

Average MAE: 0.046334012908511794
Average RMSE: 0.05785140052267175
Average Top-1 Accuracy: 0.9
Average Top-5 Accuracy: 0.9600000000000002

Average MAE: 0.017980418850504797
Average RMSE: 0.022920027030189076
Average Top-1 Accuracy: 1.0
Average Top-5 Accuracy: 1.0

Average MAE: 3.842267976142466e-06
Average RMSE: 4.893163374052718e-06
Average Top-1 Accuracy: 1.0
Average Top-5 Accuracy: 1.0

Average MAE: 3.8274386548437175e-06
Average RMSE: 4.879203707754805e-06
Average Top-1 Accuracy: 1.0
Average Top-5 Accuracy: 1.0

Average MAE: 3.830007469514384e-06
Average RMSE: 4.880158273937066e-06
Average Top-1 Accuracy: 1.0
Average Top-5 Accuracy: 1.0