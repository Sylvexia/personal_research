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

# 1-13 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Can lower to gpt-2, segfault at runtime
- Can run `resnet`
- Fix some operations

---
# The current issue

- How to debug a pass?
	- Observing log does not always work.
		- `loc("onnx.NonMaxSuppression"("NonMaxSuppression_683")): error: failed to legalize unresolved materialization from 'f32' to 'i32' that remained live after conversion.)`
	- We may do the most basic filtering to eliminate the possibilities.

---
# Log Code

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
# Log 

```cpp
Saw operation: math.exp
Saw operation: math.floor
Saw operation: math.log
Saw operation: math.tanh
Saw operation: memref.alloc
Saw operation: memref.alloca
Saw operation: memref.dim
Saw operation: memref.load
Saw operation: memref.reinterpret_cast
```

---
# gpt-2

- Lower `krnl.memcpy`, `memref.dim`, `memref.dim`
- fixed `memref` allocation dynamic shape
	- `%0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>`
- Can lower it to `.so`
- Runtime Error: `SegFault`

---
# Can run model

- `resnet18-v1-7` 
- `posit<32, 2>`
- `input: [1x3x224x224xfloat32]`
- `output: [1x1000xfloat32]`
```cpp
Running inference ...
1st iteration, 4973.498892343603, seconds
```

---
# Model cannot run:

- yolov4: `fptosi`, `math.floor`, `math.log`
- ssd-10: `arith.maxnumf`, `arith.minnumf`

---

# Future Works

- Make more model running
- Lower more required operation.