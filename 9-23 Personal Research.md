# Summary

- Complete float32 to any Posit on Python
	- 
- A lot of Posit implementation rely on keeping negative raw data as two's complement, and do another 2's complement back when doing arithmetic.

# Posit

- For Universal library. The color print has extra 2's complement only at the fraction part, in posit terminal tool, which cause it deviate from the posit standard.
- If it's negative, the whole raw bit would be real 2's complement. When `decode()` that for posit arithmetic, it would `extract_fields()` and extract the sign, exponent, fraction. The negative would do another 2's complement back to it should be.
- A lot of increase implementation does not consider when carry over the exponential. This also happened in rounding when convert from the floating point to the posit.

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

- quantize_weight
- quantize_activation

`__quantize_inputs` -> `quantize_initializer` ->`quantize_initializer_impl` -> `quantize_data` -> `quantize_nparray`

1. optional segment embedding: ??
2. Check `initializer`
	1. per-channel, standard
3. elif Check tensor and not `initializer`
4. elif parent

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
	- Initializer is basically model weight and bias

# ONNX-MLIR Frontend (Type Injection)

Summary:
Need to implement
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

