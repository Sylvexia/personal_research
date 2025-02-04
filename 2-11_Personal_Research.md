- run with only command
- compile and run separated

curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mnist-7.tar.gz --output /home/sylvex/onnx-mlir/mnist-7.tar.gz

/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v

 /home/sylvex/onnx-mlir/utils/RunONNXModelPosit.py --compile-args=-O0 --verify=ref --verify-every-value --load-ref=/home/sylvex/models/mnist-7/model/test_data_set_0 --model=/home/sylvex/models/mnist-7/model/model.onnx --print-output --n-bit=32 --es=2