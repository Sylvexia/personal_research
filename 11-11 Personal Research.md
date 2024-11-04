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