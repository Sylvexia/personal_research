Task:
5. How to implement a pass? For tablegen, how do i know what to implement.
	1. https://www.youtube.com/watch?v=UP-LBRbvI_U

what is casting in `mlir`?
- When getting the operation
	- For example, `AddOp` is derived from operation
	- We cast the operation to decide if it's `AddOp`
		- cast<>

onnx @run_main_graph

llvm
quantizeFloatToInt

tpu-mlir
```cpp
quant::UniformQuantizedType getUniformQuantizedType(Value v) {
  return v.getType()
      .cast<RankedTensorType>()
      .getElementType()
      .cast<quant::UniformQuantizedType>();
}
```

AddLowering::LoweringINT8

for operator respectively?

get values from dense element
https://discourse.llvm.org/t/using-mlir-getvalues-with-f16/3953/5

opRewritePattern v.s opConversionPattern

```cpp
struct SimplifyAddFOpPattern : public OpRewritePattern<AddFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddFOp op, PatternRewriter &rewriter) const override {
    // Assume we match some pattern and want to replace the operation.
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Value newOp = rewriter.create<SomeOtherOp>(op.getLoc(), lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
```

```cpp
struct ConvertAddFOpPattern : public OpConversionPattern<AddFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    AddFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    
    // Assume we're converting from floating-point to integer.
    auto newType = rewriter.getIntegerType(32); // target type
    Value lhs = adaptor.getOperands()[0]; // Converted operands
    Value rhs = adaptor.getOperands()[1];

    // Create a new integer add operation with the converted types.
    Value newAdd = rewriter.create<SomeIntegerAddOp>(op.getLoc(), newType, lhs, rhs);
    rewriter.replaceOp(op, newAdd);
    return success();
  }
};
```
# Convert all Arith::Const F32 to UINT32

https://www.jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/

-debug to list the pattern convert

https://mlir.llvm.org/docs/DialectConversion/#type-conversion


# Arith to Posit Function Call experiment:

- Goal: Map Arith Dialect to Posit Function Call
- Currently we only support the add and const operator, other operator should be like wise.
# MLIR Conversion Concepts

