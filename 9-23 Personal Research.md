洪祐鈞
# Summary

- Complete float32 to any Posit on Python part.
- A lot of Posit implementation rely on keeping negative raw data as two's complement, and do another 2's complement back when doing arithmetic.
- Implementing ONNX Posit converter
	- Requirement:
		- not only required to handle initializer (weight & bias)
		- need to handle the input argument and more.
		- which requires to get `MLIR` attribute right. 
		- Then bridge the tablegen generation tool to have appropriate interface. 
		- Implement the operator manually.
	- Convert the initializer Probably trivial:
		- Not all float stored initializer, we need to adapt.
- The above requires a lot of works to do
	- I still don't have idea how to get the posit interface right.
	- I have a hard time try to program the posit dialect operation interface last week.
- Key Technology first, currently too much speculation.
	- Key concept of the project:
		- Based on the type, mapping it's operation to {library call/equivalent dialect operation}
		- Goal:
			- Posit type, universal {function call/arith dialect}.
		- Mid Goal:
			- Float32 type, universal {function call}
		- Currently:
			- Float type, `libm`
				- Hopefully I can lower it such that make it executable.
		- Note: 
			- This means there would not be posit model here. We only convert float one by one to posit arithmetic.
- Million Dollar Question
	- If I have a symbol in LLVM/MLIR, how do I link to the external library?
# Posit Converter

## Value Conversion

Probably complete:

`def float32_to_posit(nbits: int, es: int, fval: float, scale: float = 1.0, saturate: bool = True) -> int:`

`python test_float32_to_posit.py`

Output:

```cpp
====================================
current float: 0.5
binary: 0b111111000000000000000000000000
nbits: 8, es: 0, scale: 1.0, saturate: True
converted posit int32: 32
converted posit binary: 0b100000
====================================
====================================
current float: -32.5
binary: 0b11000010000000100000000000000000
nbits: 16, es: 1, scale: 1.0, saturate: True
converted posit int32: 62480
converted posit binary: 0b1111010000010000
====================================
====================================
current float: -31.415926
binary: 0b11000001111110110101001111010001
nbits: 32, es: 2, scale: 1.0, saturate: True
converted posit int32: 3822755464
converted posit binary: 0b11100011110110101001111010001000
====================================
====================================
current float: 69.42069
binary: 0b1000010100010101101011101100101
nbits: 32, es: 2, scale: 1.0, saturate: True
converted posit int32: 1750514469
converted posit binary: 0b1101000010101101011101100100101
====================================
====================================
current float: 65535
binary: 0b1000111011111111111111100000000
nbits: 16, es: 2, scale: 1.0, saturate: True
converted posit int32: 31744
converted posit binary: 0b111110000000000
====================================
```
# Posit Implementation

- For Universal library. The color print has extra 2's complement only at the fraction part, in posit terminal tool, which cause it deviate from the posit standard.
- If it's negative, the whole raw bit would be real 2's complement. When `decode()` that for posit arithmetic, it would `extract_fields()` and extract the sign, exponent, fraction. The negative would do another 2's complement back to it should be.
- A lot of increase implementation does not consider when carry over the exponential. This also happened in rounding when convert from the floating point to the posit.
	- say 10011111 -> 10001000

```cpp
int main() {
  using namespace sw::universal;
  float float_num1 = -6.9;
  float float_num2 = 3.2;

  posit<16, 2> p_num1(float_num1);
  std::cout << "posit" << to_binary(p_num1) << "\n";

  posit<16, 2> p_num2(float_num2);
  auto p_res = p_num1 + p_num2;
  std::cout << "res posit" << to_binary(p_res) << "\n";

  return 0;
}
```

```bash
---------------------- CONVERT -------------------
float -6.9 sign 1 scale 2 23b fraction 0x5ccccd _fraction b10111001100110011001101
------------------- CONVERT ------------------
sign -1 scale   2 positFraction 10111001100110011001101
posit0b1.10.10.10111001101
---------------------- CONVERT -------------------
float 3.2 sign 0 scale 1 23b fraction 0x4ccccd _fraction b10011001100110011001101
------------------- CONVERT ------------------
sign  1 scale   1 positFraction 10011001100110011001101
raw bits: 1010101000110011 posit bits: 1|10-------------|10|1011100110-
raw bits: 0100110011001101 posit bits: 0|10-------------|01|1001100110-
sign -1 scale   2 r1       110111001101000
sign  1 scale   2 r2 orig  011001100110100
sign  1 scale   2 r2       100110011001100
sign -1 carry   1 sum     1011101100110100
sign -1 scale   1 sum     1101100110100000
------------------- CONVERT ------------------
sign -1 scale   1 positFraction 1101100110100000
res posit0b1.10.01.11011001101
```

# quantize

In `onnxruntime`, it's separated as these 2.
- `quantize_weight`
- `quantize_activation`

Expected code path
`__quantize_inputs` -> `quantize_initializer` ->`quantize_initializer_impl` -> `quantize_data` -> `quantize_nparray`

1. optional segment embedding: ??
2. Check `initializer`
	1. per-channel, standard
3. elif Check tensor and not `initializer`
4. elif parent
5. ...

It's hard to solve the problem directly, so we trace from the solution.
- There's a lot of edge case for quantize, coding through it may not work.
- The simplest way is to see what model actually has.

```json
graph.initialzer

{
	"dims": [
	  "32"
	],
	"dataType": 1,
	"name": "conv1.bias",
	"rawData": "1GqCvIwZtz77Jnc584ttvZG4dL2FD02+Noe7PhHiGD4tgAY/wQMDPmyZlT5RZqa+21uxvQHWr73Sca89hJ3yPseEjbyWVDG+y1dIvmkzSb6vwVY+LgtRPbEfh73r3Jw+IRYlPt00W74uQbY9NV2avaOIUr4POqU+QuyEvhzIf7o="
}
```

```json
graph.node:

{
	"input": [
	  "/Reshape_output_0",
	  "fc1.weight",
	  "fc1.bias"
	],
	"output": [
	  "/fc1/Gemm_output_0"
	],
	"name": "/fc1/Gemm",
	"opType": "Gemm",
	"attribute": [
	  {
		"name": "alpha",
		"f": 1.0,
		"type": "FLOAT"
	  },
	  {
		"name": "beta",
		"f": 1.0,
		"type": "FLOAT"
	  },
	  {
		"name": "transB",
		"i": "1",
		"type": "INT"
	  }
	],
}
```

```json

graph.node

{
	"output": [
	  "/Constant_output_0"
	],
	"name": "/Constant",
	"opType": "Constant",
	"attribute": [
	  {
		"name": "value",
		"t": {
		  "dims": [
			"2"
		  ],
		  "dataType": 7,
		  "rawData": "//////////9ADAAAAAAAAA=="
		},
		"type": "TENSOR"
	  }
	],
}
```
# Posit Converter

- Original plan: modify `graph.initializer`  
- Not only the initializer data type requires to covert, the following also need to converted
	- For MNIST Model
		- graph.node.attribute
			- Constant Op
				- value attribute when dataType: 1
			- Gemm Op
				- alpha, beta
					- scalar multiplier for `A*B`, C
- Summary:
	- Initializer is basically model weight and bias.
	- Not all float data stored in the initializer.

# ONNX-MLIR Frontend (Type Injection)

Summary:
- Need to implement mlir Attr
- Hard to get the posit attribute right at mlir side e.g. `P8E0Attr

The following is the onnx-mlir tablegen generation tool code snippet:

```python
def onnx_attr_type_to_mlir_attr_type(t):
    onnx_attr_type = Text(t)
    onnx_attr_type = onnx_attr_type[onnx_attr_type.rfind(".") + 1 :].lower()

    if onnx_attr_type == "int":
        mlir_attr_type = "SI64Attr"
    elif onnx_attr_type == "float":
        mlir_attr_type = "F32Attr"
    elif onnx_attr_type == "ints":
        mlir_attr_type = "I64ArrayAttr"
    elif onnx_attr_type == "floats":
        mlir_attr_type = "F32ArrayAttr"
    elif onnx_attr_type == "string":
        mlir_attr_type = "StrAttr"
    elif onnx_attr_type == "strings":
        mlir_attr_type = "StrArrayAttr"
    elif onnx_attr_type in {"type", "type_proto"}:
        # 'type' is the attribute type used in special_attr_types,
        # 'type_proto' is Optional op's type attribute's type
        mlir_attr_type = "TypeAttr"
    else:
        mlir_attr_type = "AnyAttr"
    # TODO: tensor and sparse tensor.
    return mlir_attr_type
```

# Map the libm

- Summary:
	- Kind of can lower math dialect to func dialect.
		- Which may further lower to llvm.
	- By doing this, it would make me learn how to do mlir conversion pass.
		- register, match and rewrite, get the operation...
	- Trying to make it runnable.
		- If I can compile with -lm and get the output.
		- Then if I have a posit wrapper:
			- uint32 posit32_add(uint8 es, uint32 a, uint32 b)
			- Map @posit32_add symbol and run it.

```cpp
func.func @exp_caller(%float: f32, %double: f64) -> (f32, f64) {
  %float_result = math.exp %float : f32
  %double_result = math.exp %double : f64
  return %float_result, %double_result : f32, f64
}
```

`./onnx-mlir-opt /home/sylvex/onnx-mlir/src/Conversion/MathToLibM/test.mlir --convert-custom-math-to-func`

```cpp
module {
  func.func private @exp(f64) -> f64 attributes {llvm.readnone}
  func.func private @expf(f32) -> f32 attributes {llvm.readnone}
  func.func @exp_caller(%arg0: f32, %arg1: f64) -> (f32, f64) {
    %0 = call @expf(%arg0) : (f32) -> f32
    %1 = call @exp(%arg1) : (f64) -> f64
    return %0, %1 : f32, f64
  }
}
```

I don't know if this would actually works or...