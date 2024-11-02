# Task

- Writing a pass that convert all `f32` data type to say, `uint8`
	- Probably should convert operation 1 by 1.
- Turn off the constant propagation in posit.
- `@run_main_graph`
	- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.

No issue:
`./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=0' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_krnl.mlir`

./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=0' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_memref.mlir

Try to get work:
`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`

# Affine

See how krnl.iterate works to get affine.
at test/mlir/krnl

# addKrnlToAffinePasses


`replaceOpWithNewOp = create<OpTy> + replaceOp(op, newOp)`

`replaceOp = replaceAllOpUsesWith + erase(Op)`

`replaceAllOpUsesWith` = `notifyOperationReplaced` + `replaceAllUsesWith`

`replaceAllUsesWith(ValueRange from, ValueRange to)` :iterate the from and to 1 by same index and `modifyOpInPlace`

`modifyOpInPlace`: Get Uses from 

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