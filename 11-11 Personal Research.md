# Task

- Writing a pass that convert all `f32` data type to say, `uint8`
	- Probably should convert operation 1 by 1.
- Turn off the constant propagation in posit.
- `@run_main_graph`
	- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.

# Affine For

op.getBody()->getArgument()

# Affine

See how krnl.iterate works to get affine.
locate at test/mlir/krnl

Try to get work:
`./onnx-mlir --EmitMLIR /home/sylvex/mnist_export/mnist_model.onnx -o ./log.txt`

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