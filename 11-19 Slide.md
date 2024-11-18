---
marp: true
theme: default
paginate: true
header: 
footer: 
style: "h1, h2, h3 {\r  text-align: center;\r}

  pre, code {\r  background-color: #ffffff;\r    \r  color: #2d2d2d; \r  \r  font-size: auto;\r }\r

  section {\r  font-size: auto;\r}\r

  img[alt~=\"center\"]\ 

  {\r  display: block;\r  margin: 0 auto;\r}"
---

# 11-19 Personal Research
## Presenter: Yu Chun Hung
## Advisor: Peng-Sheng Cheng

---

# Summary

- Successfully lowered:
	- `AffineFor`:
		- Continue from last week, we force to convert the `iterArg` type
	- `Arith` Operators:
		- Templated way to force `f32` to target int type 
			- `addf`, `subf`, `mulf`, `divf`, `select`
		- Handled `cmpf` predicate with hash map.
	- Can lower the `MNIST` model with our custom pass without error.

---

# Summary

- Future Works:
	- Integrate the custom pass to ONNX compiler.
	- In Universal wrapper library,  handle the `select`, `cmpf` operator.

---

# The `forOp` fix:

- Basically after replacing the Operation and the body...
  - we force to convert the `iterArg` type:
```cpp
auto newIterArgs = newForOp.getRegionIterArgs();
for (auto &arg : newIterArgs) {
  auto newArgType = getTypeConverter()->convertType(arg.getType());
  if (!newArgType)
	return failure();
  arg.setType(newArgType);
}
```
---

# Existing way

- Async dialect is like IREE stream dialect, it's for scheduling and synchronization.
- `AsyncToLLVM.cpp`:

```cpp
class ConvertExecuteOpTypes : public OpConversionPattern<ExecuteOp> {
public:
  LogicalResult matchAndRewrite(ExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ExecuteOp newOp =
        cast<ExecuteOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());
                                
    newOp->setOperands(adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
      return failure();
    for (auto result : newOp.getResults())
      result.setType(typeConverter->convertType(result.getType()));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
```
---

# Existing way

- Methods:
	- Clone the operation without region
	- Override new op region with old region
	- Convert and set operands and result type.
- The similarity is create new op and convert the new op afterwards.

---

# Two Arith Ops: `cmpf`, `select`

- `cmpf`: equal, not equal, greater, less,....
	- Using predicate symbol to decide how to compare.

| Symbol      | Value | String | Symbol     | Value | String |
| ----------- | ----- | ------ | ---------- | ----- | ------ |
| AlwaysFalse | `0`   | false  | UEQ        | 8     | ueq    |
| OEQ         | `1`   | oeq    | UGT        | 9     | ugt    |
| OGT         | `2`   | ogt    | UGE        | 10    | uge    |
| OGE         | `3`   | oge    | ULT        | 11    | ult    |
| OLT         | `4`   | olt    | ULE        | 12    | ule    |
| OLE         | `5`   | ole    | UNE        | 13    | une    |
| ONE         | `6`   | one    | UNO        | 14    | uno    |
| ORD         | `7`   | ord    | AlwaysTrue | 15    | true   |

---

# Two Arith Ops: `cmpf`, `select`

- `select`: base on `condition`, if true return `a` else `b`

## PseudoCode

- `CmpfOp`: `bool_res = predicate(x, y) -> bool`
- `SelectOp`: `res = (cond) ? a : b`

## MLIR Form

```cpp
%cond = arith.cmpf ogt, %x, %y : f32 // this return i1 (boolean)
%res = arith.select %cond, %a, %b : f32
```

---

# Templated Arith Lowering

- Templated way to convert with op type, op string and posit config.
```cpp
  auto populateArithBinOpPositPatterns 
    = [&](auto opType, const std::string &opString) {
    populateArithBinOpPositPattern<decltype(opType)>(
        patterns, typeConverter, opString, _n_bits, _es_val);
  };

  populateArithBinOpPositPatterns(arith::AddFOp{}, "add");
  populateArithBinOpPositPatterns(arith::SubFOp{}, "sub");
  populateArithBinOpPositPatterns(arith::MulFOp{}, "mul");
  populateArithBinOpPositPatterns(arith::DivFOp{}, "div");
  populateArithBinOpPositPatterns(arith::SelectOp{}, "select");
```

---

# CmpOps Lowering

- Currently not in templated way since the result is force to be i1 (Boolean)
- Map predicate `enum` to generated function string
- Let posit wrapper deal with the operation based on the generated function string.

---

# CmpOps Lowering

- Fun fact: SPIRV dialect (Dialect for parallel computation like GPU) has the following logic:
- `arithtoSPIRV.cpp`:
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
      // Unordered...
```

---

# Lowered result:

input:
```cpp
func.func @test_affineForLoop(%arg0: memref<64xf32>) -> f32 {
  %cst_0 = arith.constant 0.0 : f32
  %0 = affine.for %arg1 = 0 to 64 iter_args(%arg2 = %cst_0) -> (f32) {
    %arg3 = affine.load %arg0[%arg1] : memref<64xf32>
    %arg4 = arith.addf %arg3, %arg2 : f32
    %arg5 = arith.subf %arg4, %arg3 : f32
    %arg6 = arith.mulf %arg3, %arg5 : f32
    %arg7 = arith.divf %arg6, %arg3 : f32
    affine.yield %arg7 : f32
  }
  %cmp = arith.cmpf ogt, %0, %cst_0 : f32
  %select = arith.select %cmp, %0, %cst_0 : f32
  return %select : f32
}
```

---

# Lowered result:

output
```cpp
  func.func private @posit8es3_select(i1, i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_ogt(i8, i8) -> i1 attributes {llvm.readnone}
  func.func private @posit8es3_div(i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_mul(i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_sub(i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_add(i8, i8) -> i8 attributes {llvm.readnone}
  
  func.func @test_affineForLoop(%arg0: memref<64xi8>) -> i8 {
    %c0_i8 = arith.constant 0 : i8
    %0 = affine.for %arg1 = 0 to 64 iter_args(%arg2 = %c0_i8) -> (i8) {
      %3 = affine.load %arg0[%arg1] : memref<64xi8>
	  %4 = func.call @posit8es3_add(%3, %arg2) : (i8, i8) -> i8
      %5 = func.call @posit8es3_sub(%4, %3) : (i8, i8) -> i8
	  %6 = func.call @posit8es3_mul(%3, %5) : (i8, i8) -> i8
      %7 = func.call @posit8es3_div(%6, %3) : (i8, i8) -> i8
      affine.yield %7 : i8
    }
    %1 = call @posit8es3_ogt(%0, %c0_i8) : (i8, i8) -> i1
    %2 = call @posit8es3_select(%1, %0, %c0_i8) : (i1, i8, i8) -> i8
    return %2 : i8
  }
```

---

# Future Works:

- Integrate the custom pass to ONNX compiler.
- In Universal wrapper library,  handle the `select`, `cmpf` operator.