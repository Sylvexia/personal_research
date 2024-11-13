
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

arith.cmpf

- entry: `func.func @main_graph(%arg0: memref<1x1x28x28xf32>`-> `(memref<1x10xf32> {onnx.name = "19"})`
	- `attributes {llvm.emit_c_interface}`
- `"krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22x.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \2219\22 }\0A\0A]\00"} : () -> ()`