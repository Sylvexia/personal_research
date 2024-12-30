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

# 12-31 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- `onnx-mlir`
	- lowering support of math
		- `sitofp`, `exp`, `sqrt`, `tanh`, `erf`
	- test pipeline prove of concept
		- posit config, model download and run and compare numerically.
- `PositWrapper`
	- `sitofp`, `exp`, `sqrt`, `tanh`, `erf` support
	- `numpy` support in `uint` $\leftrightarrow$ and `double`
	- make install support
	- Verified C++ and python runtime can run in dependent environment.
- Currently, still, only can run `mnist` model.

---
# ONNX-MLIR Lowering

- `arith.sitofp` need to specially supported, currently only support i32 for signed integer
- Unary math is just map to function call, unary and binary are the same rule as `arith` ops.

```
  populateReturnPositOpPatterns(arith::AddFOp{}, "add");
  populateReturnPositOpPatterns(arith::SubFOp{}, "sub");
  populateReturnPositOpPatterns(arith::MulFOp{}, "mul");
  populateReturnPositOpPatterns(arith::DivFOp{}, "div");
  populateReturnPositOpPatterns(math::ExpOp{}, "exp");
  populateReturnPositOpPatterns(math::SqrtOp{}, "sqrt");
  populateReturnPositOpPatterns(math::TanhOp{}, "tanh");
  populateReturnPositOpPatterns(math::ErfOp{}, "erf");
```
---
# ONNX-MLIR test pipeline PoC

- Posit config, model download and run and compare numerically.
- How do I make it work: 
	- interface with `onnx-mlir` compiler flag to our compiler modification
	- interface environment variable to link library path
	- python posit wrapper to convert between `uint` and `double`
- Need to set universal lib environment variable.
- `python ./utils/RunONNXModelZooPosit.py -c='-O0' -m='mnist-12' -l='debug' --n-bit="16" --es='2`

---
# ONNX-MLIR test pipeline PoC

```bash
[2024-12-30 23:50:26,124] DEBUG: [mnist-12] Using Posit<16,2>
Compiling the model ...
  took  0.16668939916417003  seconds.
Loading the compiled model ...
  took  0.00015329383313655853  seconds.
Reading inputs from /tmp/tmp9q0yp1fh/mnist-12/test_data_set_0 ...
  - 1st input: [1x1x28x28xfloat32]
  done.
Running inference ...
  1st iteration, 0.6051625721156597, seconds
The 1st output Plus214_Output_0:[1x10xfloat64] is:
 [[-37.          -5.54492188  24.59375     62.71875     35.4375
   33.8125     -29.0625      37.78125    -44.96875    -71.5625    ]]
Verifying value of Plus214_Output_0:[1, 10] using atol=0.01, rtol=0.05 ...
  correct.
1 models tested: mnist-12
1 models passed: mnist-12
```

---

# Oh no only MNIST can be run:

- mnist-7 works (`softmax` -> `exp` version)
- gpt-2 
	- `reinterpret_cast` lower issue
- resnet101-v2-7 
	- cannot because has unknown dimension
- yolov4
	- `loc("StatefulPartitionedCall/model/lambda_71/Tanh"): error: failed to legalize unresolved materialization from 'f32' to 'i32' that remained live after conversion`
- ssd-10
	- `loc("onnx.NonMaxSuppression"("NonMaxSuppression_683")): error: failed to legalize unresolved materialization from 'f32' to 'i32' that remained live after conversion`

---

# `PositWrapper`

- Math function: `exp`, `sqrt`, `tanh`, `erf`
	- Just call to universal library, which call to standard c math function call.
	- Future implementation needed
	- Test: templated approach:
		- `testUnaryCFunction<8, 2, uint8_t>(256, posit8es2_exp, exp);`
		- Haven't done `arith`
- exp operation issue
	- toward negative infinity e.g. : $\exp^{-100}$, causing underflow
	- standard: 0, posit: 1
	
```
`FAIL: testInput = 11110101 positResultRaw = 00000001 doubleResultRaw = 00000000 doubleValue = -1536 doubleResult = 0`
```

---
# Universal lib exp issue

```
template<unsigned nbits, unsigned es>
posit<nbits,es> exp(posit<nbits,es> x) {
	if (isnar(x)) return x;
	posit<nbits, es> p;
	double d = std::exp(double(x));
	if (d == 0.0)
		p.minpos();//should be zero?
	else
		p = d;
	return p;
}
```

---
# `PositWrapper` `make install` support

- assume install in `custom_posit` directory.
```bash
custom_posit/
├── include
│   ├── positWrapperC.h
│   └── universal/...
└── lib
    ├── libpositWrapperC.so
    └── libpositWrapperPy.so
```
---

# `PositWrapper` `numpy support`

- Why do I spend time on this?
- Our original API has python internal conversion issue.
	- e.g. `doubleVal = posit.getDouble_8_2(rawBit)`
	- In reality its `float32` $\leftarrow$ `f(int32)`, when `i8` highest significant bit is 1, it would be interpret as negative and throw type error.
- Temporary solution: move the casting inside `c++` side
	- `numpy` array support in `uint` $\leftrightarrow$ and `double`
	- Any dimension is supported
---
# `numpy` API

- API:
```python
array = np.array([[1, 2, 4, 8, -32], [1, 2, 4, 8, -255]], dtype=np.int8)
doubleArray = posit.getDoubleArray_8_2(array)
print(doubleArray)
array = posit.getRawBitArray_8_2(doubleArray)
print(array)
```
- output:
```bash
[[ 5.96046448e-08  9.53674316e-07  1.52587891e-05  2.44140625e-04
  -1.60000000e+01]
 [ 5.96046448e-08  9.53674316e-07  1.52587891e-05  2.44140625e-04
   5.96046448e-08]]
[[  1   2   4   8 224]
 [  1   2   4   8   1]]
```

---
# Future works

- Revise our conversion scheme.
	- `reinterpret_cast` operator issue in gpt-2
	- I seemed to break the dynamic shape feature.
- I might bump to the newest version of `onnx-mlir` in the future.
- By the way,
	- [QPyTorch](https://github.com/minhhn2910/QPyTorch) has gpt-2 posit demo, we never be the first one.
