# Summary

- This week:
	- Based on MNIST model
		- Lowered all `memref` operation
		- `affine` `forOp` and related is not lowered

# Operation Handled

Based on MNIST model:
- `memref` dialect
	- `memref`: `loadop`, `reinterprete_cast`
- `affine`: `loadop`

Try to get work:
`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`

# Operation Not Handled

Based on MNIST model:
- affine: `forOp`, `yieldOp`
- templated basic `arith` operation
# How Do I normally modify the Operation

`replaceOpWithNewOp<NewOpTy>(op) = create<NewOpTy> + replaceOp(op, newOp)`
- Replace the old op's results with a new op's results' values, ensuring they match. 
- The original op is then erased.

`replaceOp(oldOp, newOp) = replaceAllOpUsesWith(oldOp, newOp->getResults()) + erase(oldOp)`
- Replace all result uses.

`replaceAllOpUsesWith` = `notifyOperationReplaced` + `replaceAllUsesWith`

`notifyOperationReplaced`: logger for the correspond rewriter.

`replaceAllUsesWith(ValueRange from, ValueRange to)` :
- Redirects any references from the old value to the new one
- Iterate the from `results` and to `results` by same index 
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

# Should we do Quantize in compiler stack?

[more quantize feature in quant dialect RFC](https://discourse.llvm.org/t/rfc-add-suport-for-quantilequantizedtype-in-quant-dialect/80346)

```
In newer work, I wouldn’t implement this concept at all in MLIR or the compiler proper but in the frontend. There are several examples of convergent evolution on this kind of thing, which show some of the different/related approaches:

- [Pytorch ao 1](https://github.com/pytorch/ao)
- [Sharktank direct quantization 7](https://github.com/nod-ai/sharktank/blob/main/docs/quantization.md) (my group develops this for certain of our optimized models)
- [Modular quant encoding 6](https://docs.modular.com/max/api/maojo/graph/quantization/)

The thing that all of these have in common is that they deal with encoding/layout/quantization far up the stack with runtime vs compile time parameterization. And they break the problem down at the top vs trying to preserve strong typing of a specific quantization algorithm deep in the compiler type hierarchy.
```

- What we can summarize from last week 'quantization' is that:
	- The quantization abstraction is done with compile time type.
- Summary of people initiate the quant dialect:
	- For the projected listed above, we can see similar evolution that implement quantization at the frontend instead of compiler stack.
	- It's more favorable parametrize at runtime but compile time.
- Is it worth it to preserve the quantization abstraction?
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