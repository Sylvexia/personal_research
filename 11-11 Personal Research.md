# Task

- Writing a pass that convert all `f32` data type to say, `uint8`
	- Probably should convert operation 1 by 1.
- Turn off the constant propagation in posit.
- `@run_main_graph`
	- `KrnlEntryPointOpLowering`
- Universal Wrapper
	- `NaR` handling.

# AffineForOp

op.getBody()->getArgument()
`./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=16 es-val=3' /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_affine.mlir --mlir-print-op-generic`

```cpp
  "func.func"() <{function_type = (i16) -> (), sym_name = "test_affineForLoop"}> ({
  ^bb0(%arg0: i16):
    %0 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1x64x14x14xi16>
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg1: index):
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg2: index):
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg3: index):
          "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
          ^bb0(%arg4: index):
            "affine.store"(%arg0, %0, %arg2, %arg3, %arg4, %arg1) <{map = #map}> : (i16, memref<1x64x14x14xi16>, index, index, index, index) -> ()
            "affine.yield"() : () -> ()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()


```
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