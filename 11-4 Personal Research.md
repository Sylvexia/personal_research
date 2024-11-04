# Task

- Writing a pass that convert all `f32` data type to say, `uint8`
	- Probably should convert operation 1 by 1.
- Turn off the constant propagation in posit.
- `@run_main_graph`
	- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.

# Summary

- This week:
	- Successfully lowered

# Not Complete

Based on MNIST
affine: for yield

# Complete

memref loadop, reinterprete_cast
affine loadop,

Try to get work:
`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`
# meow

op.getBody()->getArgument()

# Affine

See how krnl.iterate works to get affine.
locate at test/mlir/krnl

# How Do I normally modify the Operation

`replaceOpWithNewOp = create<OpTy> + replaceOp(op, newOp)`
- Create new op by type and replace the operation.

`replaceOp(oldOp, newOp) = replaceAllOpUsesWith(oldOp, ) + erase(oldOp)`
- 

`replaceAllOpUsesWith` = `notifyOperationReplaced` + `replaceAllUsesWith`

`notifyOperationReplaced`: logger for the correspond rewriter.

`replaceAllUsesWith(ValueRange from, ValueRange to)` :
- Redirects any references from the old value to the new one
- Iterate the from `operands` and to `operands` by same index 
- Set the correspond operand with new value.
- `make_early_inc_range`: 
	- The iterator increments immediately after dereferencing, allowing node deletion or insertion without disrupting the process, as long as the next iterator remains valid.

```cpp
void replaceAllUsesWith(Value from, Value to) {
  for (OpOperand &operand : 
    llvm::make_early_inc_range(from.getUses())) {
    Operation *op = operand.getOwner();
    modifyOpInPlace(op, [&]() { operand.set(to); });
  }
}

```

`modifyOpInPlace`: notify start and end of operation modification with callback.

`erase(Op)`: Using post order traversal to remove enclosing op one by one.

# Quantization

https://discourse.llvm.org/t/rfc-add-suport-for-quantilequantizedtype-in-quant-dialect/80346

```
In newer work, I wouldn’t implement this concept at all in MLIR or the compiler proper but in the frontend. There are several examples of convergent evolution on this kind of thing, which show some of the different/related approaches:

- [Pytorch ao 1](https://github.com/pytorch/ao)
- [Sharktank direct quantization 7](https://github.com/nod-ai/sharktank/blob/main/docs/quantization.md) (my group develops this for certain of our optimized models)
- [Modular quant encoding 6](https://docs.modular.com/max/api/mojo/graph/quantization/)

The thing that all of these have in common is that they deal with encoding/layout/quantization far up the stack with runtime vs compile time parameterization. And they break the problem down at the top vs trying to preserve strong typing of a specific quantization algorithm deep in the compiler type hierarchy.
```

# Materialization

The code are all look like the same.
```cpp
addSourceMaterialization([&](OpBuilder &builder, Type resultType,
						   ValueRange inputs,
						   Location loc) -> std::optional<Value> {
if (inputs.size() != 1)
  return std::nullopt;

return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
	.getResult(0);
});

addTargetMaterialization([&](OpBuilder &builder, Type resultType,
						   ValueRange inputs,
						   Location loc) -> std::optional<Value> {
if (inputs.size() != 1)
  return std::nullopt;

return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
	.getResult(0);
});
```