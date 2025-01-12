洪祐鈞

# TODO

- lower krnl.memcpy
- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- Get this work
	`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v`
- loc("onnx.Gather"("Gather_26")): error: 'memref.alloc' op dimension operand count does not equal memref dynamic dimension count
# Summary

- Now the config can run at project folder
- gmp-6.2.1

sylvex@sylvex-Aspire-A715-51G:~/onnx-mlir/build$ cat onnxDump.tmp | rg "krnl."
"krnl.memcpy"(%alloc_137, %reinterpret_cast_136, %c64_i64, %7934, %7932) : (memref<?x12x?x64xf32>, memref<?x?x12x64xf32>, i64, index, index) -> ()          "krnl.memcpy"(%alloc_177, %reinterpret_cast_176, %c64_i64, %7934, %7932) : (memref<?x12x?x64xf32>, memref<?x?x12x64xf32>, i64, index, index) -> ()          "krnl.memcpy"(

