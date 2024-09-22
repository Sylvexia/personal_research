洪祐鈞
# Summary of the week

- My computer broke, so on Saturday and Sunday I cannot do much.
- Linker error is due to unfamiliar of `LLVM` `tablegen`.
- Math dialect would be lowered further inside of `onnx-mlir` project.
- For converting floating point operations to posit, what subset of operation conversions needed to support is more important than creating Posit Dialect itself.
- The implementation of Posit Dialect interface might be extremely difficult to get it right.
	- I spent a lot of time doing it, still not able to figure a good way to do it.
	- I would consider to get the basic conversion going, then worried about the Posit Dialect interface
- Posit Converter Status: 
	- The value conversion of positive value is correct, however if the sign is negative, various implementation take 2's complement to exponent, regime, fraction, which is quite odd. I still need time to figure out why it is the case.
# Linker Error:

It turns out you can't just use the `tablegen` generated code to "magically" generate all the boilerplate code you need, you need to override some required implementation for say: `let hasCustomAssemblyFormat = 1` in `tablegen`.

And you ALWAYS need the following in your `.cpp` library to link the file.

```cpp
void PositDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "src/Dialect/Posit/PositOps.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "src/Dialect/Posit/PositOps.cpp.inc"
#include "src/Dialect/Posit/PositDialect.cpp.inc"
```

If you don't do this, the `llvm` will emit undefined reference error.
# Compiling Models 

All the information here is just for gathering what operation need to be converted first.
## Summary of compiling model

There are currently 3 models I have tested
- `mnist` (The most basic hand-writing recognition classification model)
	- No math dialect, just need to implement add/mul/const/cmp
- `whisper`
	- arith
		- add/sub/mul/div/const/cmp
		- sitofp
			- Cast from a value interpreted as a signed integer to the corresponding floating-point value
	- Math
		- exp, sqrt
		- erf
			- $\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}  dt$
- `gpt-2`
	- arith
		- add/sub/mul/div/const/cmp
		- sitofp
	- Math
		- exp, sqrt, tanh

Making the `whisper` and `gpt-2` model runs requires some data pre-processing. Notice we only cares about the **tensor** data, which is just some numbers.
## Whisper

### ONNX dialect

`./onnx-mlir-onnx /home/sylvex/Downloads/tiny.en-encoder.onnx --printIR | grep -o 'onnx\.[a-zA-Z]*' | sort | uniq`

```bash
onnx.Add
onnx.Concat
onnx.Constant
onnx.Conv
onnx.dim
onnx.Dim
onnx.Div
onnx.EntryPoint
onnx.Erf
onnx.LayerNormalization
onnx.MatMul
onnx.Mul
onnx.name
onnx.Reshape
onnx.Slice
onnx.Softmax
onnx.Transpose
onnx.Unsqueeze
```

`ONNX` dialect will not exist after that.
### Krnl dialect

`./onnx-mlir-krnl /home/sylvex/Downloads/tiny.en-encoder.onnx --printIR | grep -o 'krnl\.[a-zA-Z]*' | sort | uniq`

```bash
krnl.define
krnl.entry
krnl.get
krnl.global
krnl.iterate
krnl.load
krnl.loop
krnl.memcpy
krnl.memset
krnl.store
krnl.yield
```

`./onnx-mlir-affine /home/sylvex/Downloads/tiny.en-encoder.onnx --printIR | grep -o 'krnl\.[a-zA-Z]*' | sort | uniq`

```bash
krnl.entry
krnl.global
krnl.memcpy
```

Notice: 

4 main passes:
- `addONNXToMLIRPasses`
- `addONNXToKrnlPasses`
- `addKrnlToAffinePasses`
- `addKrnlToLLVMPasses`

Lower to affine does not means all `krnl` would disappear, some `krnl` ops would lower to `affine` and some would be lower to `llvm`.

---
## GPT-2

model source: https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/gpt-2

`./onnx-mlir-onnx /home/sylvex/Downloads/gpt2-10.onnx --printIR | grep -o 'onnx\.[a-zA-Z]*' | sort | uniq`

```
onnx.Add
onnx.Concat
onnx.Constant
onnx.dim
onnx.Dim
onnx.Div
onnx.EntryPoint
onnx.Expand
onnx.Gather
onnx.Gemm
onnx.LayerNormalization
onnx.MatMul
onnx.Mul
onnx.name
onnx.NonZero
onnx.Reshape
onnx.Slice
onnx.Softmax
onnx.Split
onnx.Squeeze
onnx.Sub
onnx.Tanh
onnx.Transpose
onnx.Unsqueeze
```

`./onnx-mlir-affine /home/sylvex/Downloads/gpt2-10.onnx --printIR | grep -o 'krnl\.[a-zA-Z]*' | sort | uniq`

```bash
krnl.entry
krnl.global
krnl.memcpy
```

### Math dialect

`./onnx-mlir-affine /home/sylvex/Downloads/gpt2-10.onnx --printIR | grep -o 'math\.[a-zA-Z]*' | sort | uniq`

```bash
math.exp
math.sqrt
math.tanh
```

### arith dialect

`./onnx-mlir-affine /home/sylvex/Downloads/gpt2-10.onnx --printIR | grep -o 'arith\.[a-zA-Z]*' | sort | uniq`

```
arith.addf
arith.addi
arith.andi
arith.cmpf
arith.cmpi
arith.constant
arith.divf
arith.floordivsi
arith.index
arith.mulf
arith.muli
arith.select
arith.sitofp
arith.subf
arith.subi
```
# How does Math Dialect get lowered?

How does the math dialect get lowered?

In `onnx-mlir` it's through 
1. `populateMathPolynomialApproximationPatterns` then
2. `populateMathToLLVMConversionPatterns`

## Exp

Lower to llvm dialect directly:

`ExpOpLowering = ConvertFMFMathToLLVMPattern<math::ExpOp, LLVM::ExpOp>`
## Tanh

1. Approximated

This is used in mlir:
```cpp
	LogicalResult
TanhApproximation::matchAndRewrite(math::TanhOp op,
                                   PatternRewriter &rewriter) const {
  if (!getElementTypeOrSelf(op.getOperand()).isF32())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  VectorShape shape = vectorShape(op.getOperand());

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  // Clamp operand into [plusClamp, minusClamp] range.
  Value minusClamp = bcast(f32Cst(builder, -7.99881172180175781f));
  Value plusClamp = bcast(f32Cst(builder, 7.99881172180175781f));
  Value x = clamp(builder, op.getOperand(), minusClamp, plusClamp);

  // Mask for tiny values that are approximated with `operand`.
  Value tiny = bcast(f32Cst(builder, 0.0004f));
  Value tinyMask = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OLT, builder.create<math::AbsFOp>(op.getOperand()),
      tiny);

  // The monomial coefficients of the numerator polynomial (odd).
  Value alpha1 = bcast(f32Cst(builder, 4.89352455891786e-03f));
  Value alpha3 = bcast(f32Cst(builder, 6.37261928875436e-04f));
  Value alpha5 = bcast(f32Cst(builder, 1.48572235717979e-05f));
  Value alpha7 = bcast(f32Cst(builder, 5.12229709037114e-08f));
  Value alpha9 = bcast(f32Cst(builder, -8.60467152213735e-11f));
  Value alpha11 = bcast(f32Cst(builder, 2.00018790482477e-13f));
  Value alpha13 = bcast(f32Cst(builder, -2.76076847742355e-16f));

  // The monomial coefficients of the denominator polynomial (even).
  Value beta0 = bcast(f32Cst(builder, 4.89352518554385e-03f));
  Value beta2 = bcast(f32Cst(builder, 2.26843463243900e-03f));
  Value beta4 = bcast(f32Cst(builder, 1.18534705686654e-04f));
  Value beta6 = bcast(f32Cst(builder, 1.19825839466702e-06f));

  // Since the polynomials are odd/even, we need x^2.
  Value x2 = builder.create<arith::MulFOp>(x, x);

  // Evaluate the numerator polynomial p.
  Value p = builder.create<math::FmaOp>(x2, alpha13, alpha11);
  p = builder.create<math::FmaOp>(x2, p, alpha9);
  p = builder.create<math::FmaOp>(x2, p, alpha7);
  p = builder.create<math::FmaOp>(x2, p, alpha5);
  p = builder.create<math::FmaOp>(x2, p, alpha3);
  p = builder.create<math::FmaOp>(x2, p, alpha1);
  p = builder.create<arith::MulFOp>(x, p);

  // Evaluate the denominator polynomial q.
  Value q = builder.create<math::FmaOp>(x2, beta6, beta4);
  q = builder.create<math::FmaOp>(x2, q, beta2);
  q = builder.create<math::FmaOp>(x2, q, beta0);

  // Divide the numerator by the denominator.
  Value res = builder.create<arith::SelectOp>(
      tinyMask, x, builder.create<arith::DivFOp>(p, q));

  rewriter.replaceOp(op, res);

  return success();
}
```
2. Expand:

This is not used directly, but it's provocative.

`ExpandPatterns.cpp`

```cpp
/// Expands tanh op into
/// 1-exp^{-2x} / 1+exp^{-2x}
/// To avoid overflow we exploit the reflection symmetry `tanh(-x) = -tanh(x)`.
/// We compute a "signs" value which is -1 if input is negative and +1 if input
/// is positive.  Then multiply the input by this value, guaranteeing that the
/// result is positive, which also guarantees `exp^{-2x * sign(x)}` is in (0,
/// 1]. Expand the computation on the input `x * sign(x)`, then multiply the
/// result by `sign(x)` to retain sign of the real result.
static LogicalResult convertTanhOp(math::TanhOp op, PatternRewriter &rewriter) {
  auto floatType = op.getOperand().getType();
  Location loc = op.getLoc();
  Value zero = createFloatConst(loc, floatType, 0.0, rewriter);
  Value one = createFloatConst(loc, floatType, 1.0, rewriter);
  Value negTwo = createFloatConst(loc, floatType, -2.0, rewriter);

  // Compute sign(x) = cast<float_type>(x < 0) * (-2) + 1
  Value isNegative = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, op.getOperand(), zero);
  Value isNegativeFloat =
      rewriter.create<arith::UIToFPOp>(loc, floatType, isNegative);
  Value isNegativeTimesNegTwo =
      rewriter.create<arith::MulFOp>(loc, isNegativeFloat, negTwo);
  Value sign = rewriter.create<arith::AddFOp>(loc, isNegativeTimesNegTwo, one);

  // Normalize input to positive value: y = sign(x) * x
  Value positiveX = rewriter.create<arith::MulFOp>(loc, sign, op.getOperand());

  // Decompose on normalized input
  Value negDoubledX = rewriter.create<arith::MulFOp>(loc, negTwo, positiveX);
  Value exp2x = rewriter.create<math::ExpOp>(loc, negDoubledX);
  Value dividend = rewriter.create<arith::SubFOp>(loc, one, exp2x);
  Value divisor = rewriter.create<arith::AddFOp>(loc, one, exp2x);
  Value positiveRes = rewriter.create<arith::DivFOp>(loc, dividend, divisor);

  // Multiply result by sign(x) to retain signs from negative inputs
  rewriter.replaceOpWithNewOp<arith::MulFOp>(op, sign, positiveRes);

  return success();
}
```

## Other ways (mapping to libm)

For example:

`void populatePatternsForOp(RewritePatternSet &patterns, MLIRContext *ctx, StringRef floatFunc, StringRef doubleFunc)`

`populatePatternsForOp<math::ErfOp>(patterns, ctx, "erff", "erf");`

Food for thought:

If we had `libposit`, we probably can do something like this. But how to link the library should be a problem.
# Future works

- Write simple conversion to capture `f32` to mapping posit lib call.
	- Then once we have posit dialect interface we can just capture `AnyPosit` instead.
- Floating trait and type to build dialect for posit dialect interface.
	- It might need put a lot of effort to get it right.
- `DenseElement`, how does it constructed from float data? The whole flow?