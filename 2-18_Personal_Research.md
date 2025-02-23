- run with only command
- compile and run separated

curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mnist-7.tar.gz --output /home/sylvex/onnx-mlir/mnist-7.tar.gz

/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v

 /home/sylvex/onnx-mlir/utils/RunONNXModelPosit.py --compile-args=-O0 --verify=ref --verify-every-value --load-ref=/home/sylvex/models/mnist-7/model/test_data_set_0 --model=/home/sylvex/models/mnist-7/model/model.onnx --print-output --n-bit=32 --es=2

/home/sylvex/onnx-mlir/utils/RunONNXModelPosit.py --compile-args=-O0 --verify=ref --verify-every-value --load-ref=/home/sylvex/models/mnist-7/model/test_data_set_0 --load-so=/home/sylvex/models/mnist-7/model/model.so --print-output --n-bit=16 --es=2

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

# Style Transfer

![](note_image/amber.jpg)
![](note_image/candy.jpg)
![](note_image/amber-candy.jpg)

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
# Working Model Listing

- Style Transfer:
	- candy-8
- Super Resolution:
	- super-resolution-10
- Object Classification:
	- resnet-18
- Object Detection:
	- version-RFB-640
	- tinyyolov2-8
- Emotional Recognition
	- emotion-ferplus-8: posit 32, 16 passed

# Future Works

- What experiment should we do?
	- What operators does a model have?
		- Numerical Analysis of supported operator.
			- We only have `+-*/` supported in posit
	- Lower bit and higher bit precision loss
		- Remember the ground truth precision is FP32
	- Model benchmark
		- Numerically fluctuation but argmax() works.
- Should I spend some time on get gpt-2 running?

# Exp

all posit32

candy-8
inference: 23298.175301510026, seconds
Done   1 tasks      | elapsed: 388.4min
posit 32, 2 correct

super-resolution-10
inference: 7055.291416308988, seconds
 Done   1 tasks      | elapsed: 117.6min
posit 32, 2 correct

version-RFB-640
inference: 1086.9763493991923, seconds
Done   1 tasks      | elapsed: 18.2min
posit 32, 2 correct

emotion-ferplus-8
inference: 1934.4427465500776, seconds
Done   1 tasks      | elapsed: 32.3min
posit 32, 2 correct

inference: 411.68988883588463, seconds
Done   1 tasks      | elapsed:  7.0min
posit 16, 2 correct

yolov4: posit 32 all nan
Done   1 tasks      | elapsed: 334.6min
posit32 failed

tinyyolov2-8
inference: 14767.52318693418, seconds
Done   1 tasks      | elapsed: 246.3min
