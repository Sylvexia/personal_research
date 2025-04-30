
# Summary

- Complete tinyyolov2 benchmark, all failed
	- All config of posit include 32-bit has large numerical error.
	- Implemented nms and mAP.
- Result in image classification for 50 testcase
	- MobileNetV2 posit(16,3) has no degradation in top 1 accuracy.
	- ResNet18 16-bit posit has no degradation in top 1 accuracy.


# tinyyolov2 Failed

The following is posit(32, 2) log
- When preprocess is `[0, 1]`, the numerical error is low
- Numerically, the 16-bit and 32-bit posit output is similar.
- If I didn't load the wrong model

```
mAP: 0.0
ground truth: -0.01834217458963394
posit: 0.3264209274202585
ground truth: 0.20137111842632294
posit: 0.3822804354131222
ground truth: 0.029884984716773033
posit: 0.29246659204363823
ground truth: -0.20655421912670135
posit: 0.4064514525234699
ground truth: -0.2856176197528839
posit: 0.6453288532793522
MAE: 1.883219511691525
RMSE: 2.7022513646256234
```

# Test case from onnx/model passed

The test case does not reflect real world data.

```
Average of input data: 0.02922861836850643
Max value of input data: 45.768131256103516
Min value of input data: -47.02507019042969
First 10 elements of input data:
Element 0: -5.962090015411377
Element 1: 1.327742576599121
Element 2: 11.280842781066895
Element 3: 1.8002662658691406
Element 4: 26.65134620666504
Element 5: 3.324601650238037
Element 6: 7.4986796379089355
Element 7: -8.365735054016113
Element 8: -19.43507957458496
Element 9: -13.389769554138184
```


# Future works

