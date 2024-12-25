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

# 12-25 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- C and Python Wrapper proof of concept
	- Separated universal wrapper. No longer inside universal library
	- Universal as 3rd party be cloned by submodule
	- `f32` and `uint` conversions in posit in python version for testing pipeline

---

# Flow Chart

![center h:480](note_image/Pasted%20image%2020241225013150.png)

---

# Answer to Posit Library

1. What library should have
	1. Should we full support of posit config?
		1. Most advanced posit arithmetic has `numpy` derived from `softposit`
			1. (8, 0), (16, 1), (32, 2)
		2. Heard universal has python wrapper but can't found one
2. Unified test to verify the correctness.
3. Auto compile and install
4. Documentation

2, 3, 4 currently has the wrapper previous stated, as a prototype.

---

# Where does the posit math lib code come from??

---

# Issue of the posit math lib

- The code basically copy of the implementation `fdlibm` in `netlib`
	- License of the `netlib`
- Only support 64-bit es 2, 3, 4
- Lower bit of operation have high chance not work with current implementation.
	- What approximation scheme is universal? Horner's approximation?
- Some operation does not work e.g. exp

---

# Issue of the posit math lib in exp operation

```cpp
Posit64 y, hi, lo, c, t;
__int32_t k = 0,xsb;
__uint32_t hx;

if(x>1) return exp(x-1) * exp(1);
else if(x<-1) return exp(x+1) / exp(1);

GET_HIGH_WORD(hx,x);
```

- Directy call when $|x| > 1$
- `GET_HIGH_WORD(hx,x);` only works for 64 bit.

---

# Contribution

- An MLIR-based compilation flow to support the evaluation of posit numbers in neural network model inference.
- Experiment: evaluate at least 2 or 3 NN models.
	- Numerical Error and E2E
	- If we can some operation in math, we might be the first team run gpt-2 with posit.
- A detail description related to the solutions for using posit numbers in MLIR, including their advantages and disadvantages.
  
---

# Test Pipeline Current Status

- What we had (not done by us):
	- Specifying the model and automatically pull from `github`
		- Both model and dataset.
	- Automatically compile and run with python runtime.
	- Able to feed the compile flag.
- Currently ongoing
	- Python Wrapper (just completed)
	- Bring the custom wrapper to compile command line.

---

# Introducion PositWrapper
## Supports C and Python

---

# Python Binding
## Input
```python
import libpositWrapperPy as posit

print("posit(64, 0)")
rawBit = posit.getRawBit_64_0(1e+307)
print(bin(rawBit))
doubleVal = posit.getDouble_64_0(rawBit)
print(doubleVal)
```
`$ python test.py`

---

## output

```bash
posit(64, 0)
0b111111111111111111111111111111111111111111111111111111111111111
4.611686018427388e+18
posit(64, 1)
0b111111111111111111111111111111111111111111111111111111111111111
2.1267647932558654e+37
posit(64, 2)
0b111111111111111111111111111111111111111111111111111111111111111
4.523128485832664e+74
posit(64, 3)
0b111111111111111111111111111111111111111111111111111111111111111
2.0458691299350887e+149
```

---

# How to add support for different posit config?

- Just add the loggic into `bind_posit_functions`, which would automatically generate the function

```cpp
// libpositWrapperPy matches the name of the .so file
PYBIND11_MODULE(libpositWrapperPy, m) {
    bind_posit_functions<8, 0, uint8_t>(m, "8_0");
    bind_posit_functions<8, 1, uint8_t>(m, "8_1");
    bind_posit_functions<8, 2, uint8_t>(m, "8_2");
    bind_posit_functions<8, 3, uint8_t>(m, "8_3");
```

---

# Symbol name

```bash
nm libpositWrapperPy.so | grep "getRawBit"

000000000006301f W _Z9getRawBitILm16ELm1EtET1_d
0000000000063509 W _Z9getRawBitILm32ELm2EjET1_d
00000000000639f7 W _Z9getRawBitILm64ELm3EmET1_d
0000000000062b31 W _Z9getRawBitILm8ELm0EhET1_d
```

---

# Auto install as python

```bash
cd /path/to/project
pip install -U pip setuptools wheel
pip install .
```

---

# Wrapper File structure

```bash
tree -I 'PositWrapper.egg-info|build|3rd_party'
.
├── CMakeLists.txt
├── include
│   └── positWrapperC.h
├── README.md
├── setup.py
├── setup.sh
├── src
│   ├── CMakeLists.txt
│   ├── positWrapperC.cpp
│   └── positWrapperPybind.cpp
└── test
    ├── CMakeLists.txt
    ├── test.cpp
    └── test.py
```
---

# Future Works

- Make the test pipeline work
- Math lowering at `onnx-mlir` end
- Implement the math operation with different posit configuration
	- sitofp, exp, sqrt, tanh, erf
	- if we can't we can treat it double operation first

---

# Merry Christmas