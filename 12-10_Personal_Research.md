
# TODO:
- self sustained MNIST in project
- what does affine lower to? 
	- cf does not work? or lowering to the cf part.

- Mnist input: [1 , 1 , 28 , 28]

Typically, `scf` is lowered to `cf` and then lowered to some final target like LLVM or SPIR-V.

`cf.assert`: assert(i1) -> str
```mlir
cf.assert %b, "Expected ... to be true"
```
`cf.br`: br(any)
```mlir
^bb2:
  %2 = call @someFn()
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```
`cf.cond_br`
```mlir
cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
```
`cf.switch`
```mlir
cf.switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```

The last three has `BranchOpInterface`: Can it be lowered by `populateBranchOpInterfaceTypeConversionPattern`?

scf at most has `RegionBranchOpInterface`
scf has `populateSCFStructuralTypeConversionsAndLegality`

affine for and if also has `RegionBranchOpInterface`
no populate ???

Remember we have standard lowering!
```
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
```

`populateReturnOpTypeConversionPattern`
```cpp
rewriter.modifyOpInPlace(op, [&] { op->setOperands(adaptor.getOperands()); });
```

`populateBranchOpInterfaceTypeConversionPattern`
```cpp
  LogicalResult
  matchAndRewrite(BranchOpInterface op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // For a branch operation, only some operands go to the target blocks, so
    // only rewrite those.
    SmallVector<Value, 4> newOperands(op->operand_begin(), op->operand_end());
    for (int succIdx = 0, succEnd = op->getBlock()->getNumSuccessors();
         succIdx < succEnd; ++succIdx) {
      OperandRange forwardedOperands =
          op.getSuccessorOperands(succIdx).getForwardedOperands();
      if (forwardedOperands.empty())
        continue;

      for (int idx = forwardedOperands.getBeginOperandIndex(),
               eidx = idx + forwardedOperands.size();
           idx < eidx; ++idx) {
        if (!shouldConvertBranchOperand || shouldConvertBranchOperand(op, idx))
          newOperands[idx] = operands[idx];
      }
    }
    rewriter.modifyOpInPlace(
        op, [newOperands, op]() { op->setOperands(newOperands); });
    return success();
  }

```