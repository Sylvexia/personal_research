洪祐鈞/Sylvex Hung
# MLIR: 

Basic type attribute works, it can be recognized in mlir:

`posit_test.mlir`:
```bash
func.func @float_attrs_pass() {
"test.float_attrs"() {
// CHECK: float_attr = 2.000000e+00 : posit8es0
float_attr = 2. : posit8es0
} : () -> ()
"test.float_attrs"() {
// CHECK: float_attr = 2.000000e+00 : posit16es1
float_attr = 2. : posit16es1
} : () -> ()
return
}
// HEX: dense<"0x000020410000A040"> : tensor<2xposit8es0>
"test.op"() {dense.attr = dense<[10.0, 5.0]> : tensor<2xposit8es0>} : () -> ()
```
compile with `mlir` and the output:
```bash
./mlir-opt ../../mlir/test/IR/posit_test.mlir
// output
module {
  func.func @float_attrs_pass() {
    "test.float_attrs"() <{float_attr = 2.000000e+00 : posit8es0}> : () -> ()
    "test.float_attrs"() <{float_attr = 2.000000e+00 : posit16es1}> : () -> ()
    return
  }
  "test.op"() {dense.attr = dense<[3.906250e-02, 1.953130e-02]> : tensor<2xposit8es0>} : () -> ()
}
```

dense value is wrong, the main reason may regard to `initFromPosit8Es0APInt` and `convertPosit8Es0APFloatToAPInt`, because I have to implement fake implementation to pass the compilation.)

## Do we have to support those 2 function?

### Init

`APFloat` -> `IEEEFloat` -> `initFromAPInt` -> `initFromPosit8Es0APInt` -> `initFromIEEEAPInt<semantic>`
currently semantic parameter modeling does not fit.

It would mattered to those who constructed `IEEEFloat` or `APFloat` 

For MLIR parsing:

```cpp
  /// Parse a floating point value with given semantics from the stream. Since
  /// this implementation parses the string as double precision and only
  /// afterwards converts the value to the requested semantic, precision may be
  /// lost.
  ParseResult parseFloat(const llvm::fltSemantics &semantics,
                         APFloat &result) override {
    bool isNegative = parser.consumeIf(Token::minus);
    Token curTok = parser.getToken();
    SMLoc loc = curTok.getLoc();

    // Check for a floating point value.
    if (curTok.is(Token::floatliteral)) {
      auto val = curTok.getFloatingPointValue();
      if (!val)
        return emitError(loc, "floating point value too large");
      parser.consumeToken(Token::floatliteral);
      result = APFloat(isNegative ? -*val : *val);
      bool losesInfo;
      result.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
      return success();
    }

    // Check for a hexadecimal float value.
    if (curTok.is(Token::integer)) {
      std::optional<APFloat> apResult;
      if (failed(parser.parseFloatFromIntegerLiteral(
              apResult, curTok, isNegative, semantics,
              APFloat::semanticsSizeInBits(semantics))))
        return failure();

      result = *apResult;
      parser.consumeToken(Token::integer);
      return success();
    }

    return emitError(loc, "expected floating point literal");
  }

```

For this, the code would break if we parse the something like `2.124130e+01`, since it would construct `APFloat`, also the precision would break.

### Convert

For `convertIEEEFloatToAPInt`: No such thing in `MLIR`, may not effect the codegen of MLIR.

If we were to implement, spec is as follows.

`void IEEEFloat::initFromPosit8Es0APInt`
- Basically from 64-bits raw data to sign, exponent, fraction

`APInt IEEEFloat::convertPosit8Es0APFloatToAPInt`
- From `APFloat` data members (sign, exponent, fraction) to `APINT`

For onnx-model, we basically feed raw-bit like data from onnx to onnx-mlir frontend.
Instead of something like `1.0324e2`, which don't requires `APFLOAT` interface that need 

## More support?

OpStatus: add, subtract, multiply, divide...

Hoping this would not effect the codegen. Currently not found.

## Worst case scenario

We might need to code a `class Posit : APFloatBase`, implement the init and convert function with passing the custom `fltsemantic`  then if constructing the `APFloat`, we can get the type from the `fltsemantic`.
# Fun Fact

in `/mlir/test/Target/LLVMIR/llvmir.mlir`
If you translate the mlir to llvmir
`llvm.mlir.global internal @f8E5M2_global_as_i8(1.5 : f8E5M2) : i8`
would downcast to:
`// CHECK: @f8E5M2_global_as_i8 = internal global i8 62`
Currently LLVMIR seemed has no support for this. It just downcast to 8-bit-integer.

But in for bf16, you can find a lot of implementation of codegen.

for `posit8es0`:
`llvm.mlir.global internal @posit8es0_global_as_i8(1.5 : posit8es0) : i8`
convert:
`@f8E5M2_global_as_i8 = internal global i8 120`

We can summarize:
LLVM has codegen for `bf16`, for `f8E5M2` you most likely to deploy with mlir with custom pass or codegen,

## ONNX

Integrated the posit type interface and type-mapping (how we pack the data) to onnx.

`onnx.in.proto` is the most import file to add types, it generate those definition and code interface for implementing the type.

```cpp
    // Non-IEEE floating-point format based on papers
    // FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
    // 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
    // Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
    // The computation usually happens inside a block quantize / dequantize
    // fused by the runtime.
    FLOAT8E4M3FN = 17;    // float 8, mostly used for coefficients, supports nan, not inf 
    FLOAT8E4M3FNUZ = 18;  // float 8, mostly used for coefficients, supports nan, not inf, no negative zero 
    FLOAT8E5M2 = 19;      // follows IEEE 754, supports nan, inf, mostly used for gradients
    FLOAT8E5M2FNUZ = 20;  // follows IEEE 754, supports nan, inf, mostly used for gradients, no negative zero
    POSIT8ES0 = 21;
    POSIT16ES1 = 22;
```

The pain point here is for types like FLOAT8E4M3FN, there are Cast, CastLike, QuantizeLinear, DequantizeLinear get implemented.

The common operator like Conv does not accept the FLOAT8E4M3FN as input.

# ONNX-MLIR

Currently onnx-mlir frontend can accept the posit type and lower to onnx dialect.

`posit_nonraw_data.json`

```json
{
    "irVersion": "9",
    "graph": {
      "node": [
        {
          "output": [
            "output_posit8es0"
          ],
          "opType": "Constant",
          "attribute": [
            {
              "name": "value",
              "t": {
                "dims": [
                  "2"
                ],
                "dataType": 21,
                "int32Data": [
                  184,
                  116
                ],
                "name": "tensor_posit8es0"
              },
              "type": "TENSOR"
            }
          ]
        },
        {
          "output": [
            "output_posit16es1"
          ],
          "opType": "Constant",
          "attribute": [
            {
              "name": "value",
              "t": {
                "dims": [
                  "2"
                ],
                "dataType": 22,
                "int32Data": [
                  192,
                  124
                ],
                "name": "output_posit16es1"
              },
              "type": "TENSOR"
            }
          ]
        }
      ],
      "name": "fp8_nonraw_data",
      "output": [
        {
          "name": "output_posit8es0",
          "type": {
            "tensorType": {
              "elemType": 21,
              "shape": {
                "dim": [
                  {
                    "dimValue": "2"
                  }
                ]
              }
            }
          }
        },
        {
          "name": "output_posit16es1",
          "type": {
            "tensorType": {
              "elemType": 22,
              "shape": {
                "dim": [
                  {
                    "dimValue": "2"
                  }
                ]
              }
            }
          }
        }
      ]
    },
    "opsetImport": [
      {
        "version": "19"
      }
    ]
  }

```
`./onnx-mlir --EmitONNXBasic --printIR /home/sylvex/onnx-mlir/test/mlir/onnx/parse/posit_nonraw_data.json`
```
"builtin.module"() ({
  "func.func"() <{function_type = () -> (tensor<2xposit8es0>, tensor<2xposit16es1>), res_attrs = [{onnx.name = "output_posit8es0"}, {onnx.name = "output_posit16es1"}], sym_name = "main_graph"}> ({
    %0 = "onnx.Constant"() {value = #onnx.dense_disposable<1:"0xB874"> : tensor<2xposit8es0>} : () -> tensor<2xposit8es0>
    %1 = "onnx.Constant"() {value = #onnx.dense_disposable<2:"0xC0007C00"> : tensor<2xposit16es1>} : () -> tensor<2xposit16es1>
    "onnx.Return"(%0, %1) : (tensor<2xposit8es0>, tensor<2xposit16es1>) -> ()
  }) : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}) {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "posit_nonraw_data"} : () -> ()
```

The constant operator is not implemented, it kinda odd this would work.

The ONNX dialect definition tablegen is from gen_onnx_mlir.py, which is based on onnx proto defininition.

Maybe the `.onnx` should be tested?

# Reference

The following implementation in `mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp` that substitute `math.AsinOp` to `arith` operation is like:
`posit.add -> arith` operation

```mlir
namespace {
struct AsinPolynomialApproximation : public OpRewritePattern<math::AsinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AsinOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace
LogicalResult
AsinPolynomialApproximation::matchAndRewrite(math::AsinOp op,
                                             PatternRewriter &rewriter) const {
  Value operand = op.getOperand();
  Type elementType = getElementTypeOrSelf(operand);

  if (!(elementType.isF32() || elementType.isF16()))
    return rewriter.notifyMatchFailure(op,
                                       "only f32 and f16 type is supported.");
  VectorShape shape = vectorShape(operand);

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  auto fma = [&](Value a, Value b, Value c) -> Value {
    return builder.create<math::FmaOp>(a, b, c);
  };

  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };

  Value s = mul(operand, operand);
  Value q = mul(s, s);
  Value r = bcast(floatCst(builder, 5.5579749017470502e-2, elementType));
  Value t = bcast(floatCst(builder, -6.2027913464120114e-2, elementType));

  r = fma(r, q, bcast(floatCst(builder, 5.4224464349245036e-2, elementType)));
  t = fma(t, q, bcast(floatCst(builder, -1.1326992890324464e-2, elementType)));
  r = fma(r, q, bcast(floatCst(builder, 1.5268872539397656e-2, elementType)));
  t = fma(t, q, bcast(floatCst(builder, 1.0493798473372081e-2, elementType)));
  r = fma(r, q, bcast(floatCst(builder, 1.4106045900607047e-2, elementType)));
  t = fma(t, q, bcast(floatCst(builder, 1.7339776384962050e-2, elementType)));
  r = fma(r, q, bcast(floatCst(builder, 2.2372961589651054e-2, elementType)));
  t = fma(t, q, bcast(floatCst(builder, 3.0381912707941005e-2, elementType)));
  r = fma(r, q, bcast(floatCst(builder, 4.4642857881094775e-2, elementType)));
  t = fma(t, q, bcast(floatCst(builder, 7.4999999991367292e-2, elementType)));
  r = fma(r, s, t);
  r = fma(r, s, bcast(floatCst(builder, 1.6666666666670193e-1, elementType)));
  t = mul(operand, s);
  r = fma(r, t, operand);

  rewriter.replaceOp(op, r);
  return success();
}
```

# TODO:

## ONNX

`numpy_helper.py`:
`posit8es0_to_float32(data, dims)`

`helper.py`:
`float32_to_posit8es0`

The same logic can also be used to `onnx-posit-converter`, or maybe `APFloat` related interface if needed.

## ONNX-MLIR

- Check if all the data in the model is represent by raw bit data instead of float stream
	- If float stream we might need to go back to implement APFloat interface.
- Posit dialect
	- tablegen interface
	- create the op programatically to verify
## Model

- Collect the operation needed in order to compile MNIST Model.
- How to convert all tensor types in the .onnx model file? (Model converter)
## Posit

- See the universal/softposit implementation.

## Flowchart