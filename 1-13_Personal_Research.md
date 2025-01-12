洪祐鈞

# TODO

- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- Get this work
	`/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir --EmitLib --enable-posit --n-bits=32 --es-val=2 /home/sylvex/GPT2/model.onnx -o model.so -L/home/sylvex/custom_posit/lib/ -lpositWrapperC -v`
- loc("onnx.Gather"("Gather_26")): error: 'memref.alloc' op dimension operand count does not equal memref dynamic dimension count
- yolov4 fptosi, math.floor, math.log
- ssd-10 failed: arith.maxnumf, arith.minnumf 
- gmp-6.2.1
# Summary

- Can lower to gpt-2
- Now the config can run at project folder

---
# The current issue

- How to debug a pass?
	- List all the captured operations:
	- `loc("onnx.NonMaxSuppression"("NonMaxSuppression_683")): error: failed to legalize unresolved materialization from 'f32' to 'i32' that remained live after conversion.)`

```cpp
auto module = getOperation();
std::set<std::string> operationNames;
module.walk([&](Operation *op) {
  operationNames.insert(op->getName().getStringRef().str());
});

for (const auto &name : operationNames) {
  llvm::errs() << "Saw operation: " << name << "\n";
}
```

---

# 