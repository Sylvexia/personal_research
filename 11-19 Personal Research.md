# Summary

Successfully lowered:
- `AffineFor`:
	- Continue from last week, we force to convert the `iterArg` type
- `Arith` Operators:
	- Templated way to force `f32` to target int type 
		- `addf`, `subf`, `mulf`, `divf`, select
	- Handled `cmpf` predicate with hash map.
- Can lower the MNIST with our custom pass without error.

# The forOp fix:

- Basically after replace the Operation and the body, force to convert the `iterArg` type:
```cpp
auto newIterArgs = newForOp.getRegionIterArgs();
for (auto &arg : newIterArgs) {
  auto newArgType = getTypeConverter()->convertType(arg.getType());
  if (!newArgType)
	return failure();
  arg.setType(newArgType);
}
```
# Existing way

- Async dialect is like IREE stream dialect, target for scheduling and synchronization.
	- OMG it even model coroutines. Prolly useful for MLIR runtime.
- `AsyncToLLVM.cpp`:
```cpp
class ConvertExecuteOpTypes : public OpConversionPattern<ExecuteOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ExecuteOp newOp =
        cast<ExecuteOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
                                
    newOp->setOperands(adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
      return failure();
    for (auto result : newOp.getResults())
      result.setType(typeConverter->convertType(result.getType()));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
```

- Methods:
	- Clone the operation without region
	- Override new op region with old region
	- Convert and set operands and result type.
- The similarity is create new op and convert the new op afterwards.

# Two Arith Ops: `cmpf`, `select`

- Predicate table:

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
- `CmpfOp`: `bool_res = predicate(x, y) -> bool`
- `SelectOp`: `res = (cond) ? a : b`
	```cpp
	%cond = arith.cmpf ogt, %x, %y : f32 // this return bool or i1
	%res = arith.select %cond, %a, %b : f32
	```

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
# CmpOps Lowering

- Currently not in templated way since the result is force to be i1 (Boolean)
- Map predicate `enum` to generated function string
- Let posit wrapper deal with the operation based on the generated function string.
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

# Lowered result:

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

rewriter.modifyOpInPlace

- entry: `func.func @main_graph(%arg0: memref<1x1x28x28xf32>`-> `(memref<1x10xf32> {onnx.name = "19"})`
	- `attributes {llvm.emit_c_interface}`
- `"krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22x.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \2219\22 }\0A\0A]\00"} : () -> ()`

signature:??

`./onnx-mlir-opt --convert-arith-to-posit-func='n-bits=8 es-val=3' /home/sylvex/onnx-mlir/build/Debug/bin/log.txt.onnx.mlir --mlir-elide-elementsattrs-if-larger=16 > lowered.mlir`

success

```cpp
#map = affine_map<(d0, d1) -> (d0 * 32 + d1)>
#map1 = affine_map<(d0) -> (-d0 + 1, 0)>
#map2 = affine_map<(d0) -> (-d0 + 29, 3)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#map4 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#map5 = affine_map<(d0) -> (0, d0 * 2)>
#map6 = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
#map7 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
#map8 = affine_map<(d0) -> (-d0 + 15, 3)>
#map9 = affine_map<(d0, d1) -> (d0 + d1 * 32)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "log.txt"} {
  func.func private @posit8es3_ogt(i8, i8) -> i1 attributes {llvm.readnone}
  func.func private @posit8es3_select(i1, i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_oge(i8, i8) -> i1 attributes {llvm.readnone}
  func.func private @posit8es3_add(i8, i8) -> i8 attributes {llvm.readnone}
  func.func private @posit8es3_mul(i8, i8) -> i8 attributes {llvm.readnone}
  func.func @main_graph(%arg0: memref<1x1x28x28xi8> {onnx.name = "x.1"}) -> (memref<1x10xi8> {onnx.name = "19"}) attributes {llvm.emit_c_interface} {
    %c-1_i8 = arith.constant -1 : i8
    %c14 = arith.constant 14 : index
    %c0_i8 = arith.constant 0 : i8
    %c2 = arith.constant 2 : index
    %c28 = arith.constant 28 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "krnl.global"() {name = "name", shape = [32], value = dense_resource<__elided__> : tensor<32xi8>} : () -> memref<32xi8>
    %1 = "krnl.global"() {name = "name", shape = [32, 1, 3, 3], value = dense_resource<__elided__> : tensor<32x1x3x3xi8>} : () -> memref<32x1x3x3xi8>
    %2 = "krnl.global"() {name = "name", shape = [64], value = dense_resource<__elided__> : tensor<64xi8>} : () -> memref<64xi8>
    %3 = "krnl.global"() {name = "name", shape = [64, 32, 3, 3], value = dense_resource<__elided__> : tensor<64x32x3x3xi8>} : () -> memref<64x32x3x3xi8>
    %4 = "krnl.global"() {name = "name", shape = [128], value = dense_resource<__elided__> : tensor<128xi8>} : () -> memref<128xi8>
    %5 = "krnl.global"() {name = "name", shape = [128, 3136], value = dense_resource<__elided__> : tensor<128x3136xi8>} : () -> memref<128x3136xi8>
    %6 = "krnl.global"() {name = "name", shape = [10], value = dense<[41, 51, -82, 39, -77, 46, -77, -80, 48, -85]> : tensor<10xi8>} : () -> memref<10xi8>
    %7 = "krnl.global"() {name = "name", shape = [10, 128], value = dense_resource<__elided__> : tensor<10x128xi8>} : () -> memref<10x128xi8>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x32x28x28xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 32 {
          %8 = affine.apply #map(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %9 = affine.for %arg6 = 0 to 1 iter_args(%arg7 = %c0_i8) -> (i8) {
                %12 = affine.for %arg8 = max #map1(%arg4) to min #map2(%arg4) iter_args(%arg9 = %arg7) -> (i8) {
                  %13 = affine.for %arg10 = max #map1(%arg5) to min #map2(%arg5) iter_args(%arg11 = %arg9) -> (i8) {
                    %14 = affine.apply #map3(%arg6)[%arg2]
                    %15 = affine.apply #map4(%arg8, %arg4)
                    %16 = affine.apply #map4(%arg10, %arg5)
                    %17 = affine.load %arg0[%arg1, %14, %15, %16] : memref<1x1x28x28xi8>
                    %18 = affine.load %1[%8, %arg6, %arg8, %arg10] : memref<32x1x3x3xi8>
                    %19 = func.call @posit8es3_mul(%17, %18) : (i8, i8) -> i8
                    %20 = func.call @posit8es3_add(%arg11, %19) : (i8, i8) -> i8
                    affine.yield %20 : i8
                  }
                  affine.yield %13 : i8
                }
                affine.yield %12 : i8
              }
              %10 = affine.load %0[%8] : memref<32xi8>
              %11 = func.call @posit8es3_add(%9, %10) : (i8, i8) -> i8
              affine.store %11, %alloc[%arg1, %8, %arg4, %arg5] : memref<1x32x28x28xi8>
            }
          }
        }
      }
    }
    %alloc_0 = memref.alloc() {alignment = 16 : i64} : memref<1x32x28x28xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 32 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %8 = affine.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x32x28x28xi8>
            %9 = func.call @posit8es3_oge(%8, %c0_i8) : (i8, i8) -> i1
            %10 = func.call @posit8es3_select(%9, %8, %c0_i8) : (i1, i8, i8) -> i8
            affine.store %10, %alloc_0[%arg1, %arg2, %arg3, %arg4] : memref<1x32x28x28xi8>
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<1x32x14x14xi8>
    %alloca = memref.alloca() : memref<i8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 32 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            affine.store %c-1_i8, %alloca[] : memref<i8>
            %8 = affine.max #map5(%arg3)
            %9 = affine.max #map5(%arg4)
            affine.for %arg5 = 0 to min #map6(%arg3)[%c28, %c2, %c0, %c2, %c1] {
              affine.for %arg6 = 0 to min #map6(%arg4)[%c28, %c2, %c0, %c2, %c1] {
                %11 = arith.addi %arg5, %8 : index
                %12 = arith.addi %arg6, %9 : index
                %13 = memref.load %alloc_0[%arg1, %arg2, %11, %12] : memref<1x32x28x28xi8>
                %14 = affine.load %alloca[] : memref<i8>
                %15 = func.call @posit8es3_ogt(%14, %13) : (i8, i8) -> i1
                %16 = func.call @posit8es3_select(%15, %14, %13) : (i1, i8, i8) -> i8
                affine.store %16, %alloca[] : memref<i8>
              }
            }
            %10 = affine.load %alloca[] : memref<i8>
            affine.store %10, %alloc_1[%arg1, %arg2, %arg3, %arg4] : memref<1x32x14x14xi8>
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<1x64x14x14xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 64 {
          %8 = affine.apply #map7(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %9 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %c0_i8) -> (i8) {
                %12 = affine.for %arg8 = max #map1(%arg4) to min #map8(%arg4) iter_args(%arg9 = %arg7) -> (i8) {
                  %13 = affine.for %arg10 = max #map1(%arg5) to min #map8(%arg5) iter_args(%arg11 = %arg9) -> (i8) {
                    %14 = affine.apply #map9(%arg6, %arg2)
                    %15 = affine.apply #map4(%arg8, %arg4)
                    %16 = affine.apply #map4(%arg10, %arg5)
                    %17 = affine.load %alloc_1[%arg1, %14, %15, %16] : memref<1x32x14x14xi8>
                    %18 = affine.load %3[%8, %arg6, %arg8, %arg10] : memref<64x32x3x3xi8>
                    %19 = func.call @posit8es3_mul(%17, %18) : (i8, i8) -> i8
                    %20 = func.call @posit8es3_add(%arg11, %19) : (i8, i8) -> i8
                    affine.yield %20 : i8
                  }
                  affine.yield %13 : i8
                }
                affine.yield %12 : i8
              }
              %10 = affine.load %2[%8] : memref<64xi8>
              %11 = func.call @posit8es3_add(%9, %10) : (i8, i8) -> i8
              affine.store %11, %alloc_2[%arg1, %8, %arg4, %arg5] : memref<1x64x14x14xi8>
            }
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x64x14x14xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %8 = affine.load %alloc_2[%arg1, %arg2, %arg3, %arg4] : memref<1x64x14x14xi8>
            %9 = func.call @posit8es3_oge(%8, %c0_i8) : (i8, i8) -> i1
            %10 = func.call @posit8es3_select(%9, %8, %c0_i8) : (i1, i8, i8) -> i8
            affine.store %10, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x64x14x14xi8>
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<1x64x7x7xi8>
    %alloca_5 = memref.alloca() : memref<i8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            affine.store %c-1_i8, %alloca_5[] : memref<i8>
            %8 = affine.max #map5(%arg3)
            %9 = affine.max #map5(%arg4)
            affine.for %arg5 = 0 to min #map6(%arg3)[%c14, %c2, %c0, %c2, %c1] {
              affine.for %arg6 = 0 to min #map6(%arg4)[%c14, %c2, %c0, %c2, %c1] {
                %11 = arith.addi %arg5, %8 : index
                %12 = arith.addi %arg6, %9 : index
                %13 = memref.load %alloc_3[%arg1, %arg2, %11, %12] : memref<1x64x14x14xi8>
                %14 = affine.load %alloca_5[] : memref<i8>
                %15 = func.call @posit8es3_ogt(%14, %13) : (i8, i8) -> i1
                %16 = func.call @posit8es3_select(%15, %14, %13) : (i1, i8, i8) -> i8
                affine.store %16, %alloca_5[] : memref<i8>
              }
            }
            %10 = affine.load %alloca_5[] : memref<i8>
            affine.store %10, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x64x7x7xi8>
          }
        }
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %alloc_4 to offset: [0], sizes: [1, 3136], strides: [3136, 1] : memref<1x64x7x7xi8> to memref<1x3136xi8>
    %alloc_6 = memref.alloc() {alignment = 128 : i64} : memref<1x128xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        %alloca_9 = memref.alloca() : memref<i8>
        affine.store %c0_i8, %alloca_9[] : memref<i8>
        affine.for %arg3 = 0 to 3136 {
          %11 = affine.load %reinterpret_cast[%arg1, %arg3] : memref<1x3136xi8>
          %12 = affine.load %5[%arg2, %arg3] : memref<128x3136xi8>
          %13 = func.call @posit8es3_mul(%11, %12) : (i8, i8) -> i8
          %14 = affine.load %alloca_9[] : memref<i8>
          %15 = func.call @posit8es3_add(%13, %14) : (i8, i8) -> i8
          affine.store %15, %alloca_9[] : memref<i8>
        }
        %8 = affine.load %alloca_9[] : memref<i8>
        %9 = affine.load %4[%arg2] : memref<128xi8>
        %10 = func.call @posit8es3_add(%8, %9) : (i8, i8) -> i8
        affine.store %10, %alloc_6[%arg1, %arg2] : memref<1x128xi8>
      }
    }
    %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<1x128xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        %8 = affine.load %alloc_6[%arg1, %arg2] : memref<1x128xi8>
        %9 = func.call @posit8es3_oge(%8, %c0_i8) : (i8, i8) -> i1
        %10 = func.call @posit8es3_select(%9, %8, %c0_i8) : (i1, i8, i8) -> i8
        affine.store %10, %alloc_7[%arg1, %arg2] : memref<1x128xi8>
      }
    }
    %alloc_8 = memref.alloc() {alignment = 128 : i64} : memref<1x10xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 10 {
        %alloca_9 = memref.alloca() : memref<i8>
        affine.store %c0_i8, %alloca_9[] : memref<i8>
        affine.for %arg3 = 0 to 128 {
          %11 = affine.load %alloc_7[%arg1, %arg3] : memref<1x128xi8>
          %12 = affine.load %7[%arg2, %arg3] : memref<10x128xi8>
          %13 = func.call @posit8es3_mul(%11, %12) : (i8, i8) -> i8
          %14 = affine.load %alloca_9[] : memref<i8>
          %15 = func.call @posit8es3_oge(%13, %14) : (i8, i8) -> i8
          affine.store %15, %alloca_9[] : memref<i8>
        }
        %8 = affine.load %alloca_9[] : memref<i8>
        %9 = affine.load %6[%arg2] : memref<10xi8>
        %10 = func.call @posit8es3_add(%8, %9) : (i8, i8) -> i8
        affine.store %10, %alloc_8[%arg1, %arg2] : memref<1x10xi8>
      }
    }
    return %alloc_8 : memref<1x10xi8>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 1 , 28 , 28] , \22name\22 : \22x.1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 10] , \22name\22 : \2219\22 }\0A\0A]\00"} : () -> ()
}
```

# universal lib

@posit8es3_oge

We need to 
```cpp
template<unsigned _nbits, unsigned _es>
class posit {
...
```