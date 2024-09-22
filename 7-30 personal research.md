
src/Dialect/ONNX/ONNXOps.td.inc

```
def ONNXAddOp:ONNX_Op<"Add",

[Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

let hasCanonicalizer = 1;

let summary = "ONNX Add operation";

let description = [{

Performs element-wise binary addition (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

(Opset 14 change): Extend supported types to include uint8, int8, uint16, and int16.

}];

let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,

AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$B);

let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
```

```
def Posit8AddOp:Posit_Op<"Add",[Pure]> {

let summary = "Posit Add operation";

let arguments = (ins Posit8Attr:$A, Posit8Attr:$B);

let results = (outs Posit8Attr:$C);
}
```

./onnx-mlir /home/sylvex/mnist_export/mnist_model.onnx -debug-only=dialect-conversion &> change.txt

![[change.txt]]

```
class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {

public:

using OpConversionPattern::OpConversionPattern;

LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,

ConversionPatternRewriter &rewriter) const override {

  

Value input = adaptor.getX();

  

// Quantized types are not supported right now (in type conversion).

// Once they are, the input should be rescaled for quantized types. (TBD)

// Maps to `tosa.clamp` which has both int and fp limits.

rewriter.replaceOpWithNewOp<mlir::tosa::ClampOp>(op, op.getType(), input,

rewriter.getI64IntegerAttr(0),

rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),

rewriter.getF32FloatAttr(0.0f),

rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));

return success();

}

};
```

populateLoweringONNXElementwiseOpPattern

there's relu cos min