
# Summary

- Complete tinyyolov2 benchmark, all failed
	- All config of posit include 32-bit has large numerical error.
	- Implemented nms and mAP.
- Result in image classification
	- 


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

