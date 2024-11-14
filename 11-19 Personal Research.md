successfully lowered:
- affine For
- cmpf
- 
# What is Region/Block ?

```cpp
void modifyBlockArgumentType(FuncOp funcOp, unsigned argIndex, Type newType) {
  // Get the entry block of the function
  Block &entryBlock = funcOp.getBody().front();

  // Create a new block with the updated argument types
  OpBuilder builder(funcOp.getContext());
  Block *newBlock = builder.createBlock(&funcOp.getBody());
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    if (i == argIndex) {
      newBlock->addArgument(newType, entryBlock.getArgument(i).getLoc());
    } else {
      newBlock->addArgument(entryBlock.getArgument(i).getType(), entryBlock.getArgument(i).getLoc());
    }
  }

  // Move operations from the old block to the new block
  newBlock->getOperations().splice(newBlock->end(), entryBlock.getOperations());

  // Replace all uses of the old block argument with the new block argument
  entryBlock.getArgument(argIndex).replaceAllUsesWith(newBlock->getArgument(argIndex));

  // Erase the old block
  entryBlock.erase();
}
```

```cpp
    // op.getRegionIterArgs();
    // getBody(0)->getArguments().drop_front();

    // op.getInductionVar();
    // getBody(0)->getArgument(0);

    // op.getBody(0);
    // region[0].front()
```

inlineBlockBefore = replacealluse(block argument) + dest->getOperations().splice

# The fix:
```cpp
auto newIterArgs = newForOp.getRegionIterArgs();
for (auto &arg : newIterArgs) {
  auto newArgType = getTypeConverter()->convertType(arg.getType());
  if (!newArgType)
	return failure();
  arg.setType(newArgType);
}
```

AsyncToLLVM.cpp:

ConvertExecuteOpTypes
cloneOp
inlineRegion
setOperands
getresults
replaceOp

rewriter.modifyOpInPlace

%9 = arith.cmpf oge, %8, %cst_0 : f32

predicate:

| Symbol      | Value | String |
| ----------- | ----- | ------ |
| AlwaysFalse | `0`   | false  |
| OEQ         | `1`   | oeq    |
| OGT         | `2`   | ogt    |
| OGE         | `3`   | oge    |
| OLT         | `4`   | olt    |
| OLE         | `5`   | ole    |
| ONE         | `6`   | one    |
| ORD         | `7`   | ord    |
| UEQ         | `8`   | ueq    |
| UGT         | `9`   | ugt    |
| UGE         | `10`  | uge    |
| ULT         | `11`  | ult    |
| ULE         | `12`  | ule    |
| UNE         | `13`  | une    |
| UNO         | `14`  | uno    |
| AlwaysTrue  | `15`  | true   |

arithtoSPIRV
```cpp
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(op, adaptor.getLhs(),                 \
                                         adaptor.getRhs());                    \
    return success();

      // Ordered.
      DISPATCH(arith::CmpFPredicate::OEQ, spirv::FOrdEqualOp);
      DISPATCH(arith::CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
      DISPATCH(arith::CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
      DISPATCH(arith::CmpFPredicate::OLT, spirv::FOrdLessThanOp);
      DISPATCH(arith::CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
      DISPATCH(arith::CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
      // Unordered.

```

- entry: `func.func @main_graph(%arg0: memref<1x1x28x28xf32>`-> `(memref<1x10xf32> {onnx.name = "19"})`
	- `attributes {llvm.emit_c_interface}`
- `"krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22x.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \2219\22 }\0A\0A]\00"} : () -> ()`

`./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=3' /home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir`