
curl --insecure --retry 50 --location --silent https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-7.tar.gz --time-cond /home/sylvex/onnx-mlir/mnist-7.tar.gz --output /home/sylvex/onnx-mlir/mnist-7.tar.gz

/home/sylvex/onnx-mlir/utils/RunONNXModelPosit.py --compile-args=-O0 --verify=ref --verify-every-value --load-ref=/tmp/tmp1bxtipqm/model/test_data_set_0 --model=/tmp/tmp1bxtipqm/model/model.onnx --print-output --n-bit=32 --es=2