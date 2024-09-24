
https://discourse.llvm.org/t/mlir-how-do-i-link-an-external-c-function-for-an-operation-in-an-mlir-file/1821/2

```cpp
func.func @exp_caller(%float: f32, %double: f64) -> (f32, f64) {
  %float_result = math.exp %float : f32
  %double_result = math.exp %double : f64
  return %float_result, %double_result : f32, f64
}
```

`./onnx-mlir-opt /home/sylvex/onnx-mlir/src/Conversion/MathToLibM/test.mlir --convert-custom-math-to-func`

```cpp
module {
  func.func private @exp(f64) -> f64 attributes {llvm.readnone}
  func.func private @expf(f32) -> f32 attributes {llvm.readnone}
  func.func @exp_caller(%arg0: f32, %arg1: f64) -> (f32, f64) {
    %0 = call @expf(%arg0) : (f32) -> f32
    %1 = call @exp(%arg1) : (f64) -> f64
    return %0, %1 : f32, f64
  }
}
```

`./onnx-mlir-opt /home/sylvex/onnx-mlir/src/Conversion/MathToLibM/test.mlir --convert-custom-math-to-func > func.mlir`

`/home/sylvex/onnx_llvm/llvm-project/build/bin/mlir-opt func.mlir --convert-func-to-llvm > llvm.mlir`

```cpp
module {
  llvm.func @exp(f64) -> f64 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, sym_visibility = "private"}
  llvm.func @expf(f32) -> f32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, sym_visibility = "private"}
  llvm.func @exp_caller(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> {
    %0 = llvm.call @expf(%arg0) : (f32) -> f32
    %1 = llvm.call @exp(%arg1) : (f64) -> f64
    %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)>
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)>
    llvm.return %4 : !llvm.struct<(f32, f64)>
  }
}
```


# onnx-mlir

```cpp
  FlatSymbolRefAttr getOrInsertUnaryMathFunction(PatternRewriter &rewriter,
      ModuleOp module, std::string mathFuncName, mlir::Type llvmInType,
      mlir::Type llvmOutType) const {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(mathFuncName))
      return SymbolRefAttr::get(context, mathFuncName);

    // Create function declaration.
    // auto llvmF32Ty = FloatType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(
        llvmOutType, ArrayRef<mlir::Type>({llvmInType}));

    // Insert the unary math function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(
        module.getLoc(), mathFuncName, llvmFnType);
    return SymbolRefAttr::get(context, mathFuncName);
  }
```
# Linalg
```cpp
LogicalResult mlir::linalg::LinalgOpToLibraryCallRewrite::matchAndRewrite(
    LinalgOp op, PatternRewriter &rewriter) const {
  auto libraryCallName = getLibraryCallSymbolRef(op, rewriter);
  if (failed(libraryCallName))
    return failure();

  // TODO: Add support for more complex library call signatures that include
  // indices or captured values.
  rewriter.replaceOpWithNewOp<func::CallOp>(
      op, libraryCallName->getValue(), TypeRange(),
      createTypeCanonicalizedMemRefOperands(rewriter, op->getLoc(),
                                            op->getOperands()));
  return success();
}
```

```cpp
static FailureOr<FlatSymbolRefAttr>
getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty())
    return rewriter.notifyMatchFailure(op, "No library call defined for: ");

  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  SmallVector<Type, 4> inputTypes(extractOperandTypes(op));
  if (op->getNumResults() != 0) {
    return rewriter.notifyMatchFailure(
        op,
        "Library call for linalg operation can be generated only for ops that "
        "have void return types");
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  // Insert a function attribute that will trigger the emission of the
  // corresponding `_mlir_ciface_xxx` interface so that external libraries see
  // a normalized ABI. This interface is added during std to llvm conversion.
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(op->getContext()));
  funcOp.setPrivate();
  return fnNameAttr;
}
```

```cpp
std::string mlir::linalg::generateLibraryCallName(Operation *op) {
  assert(isa<LinalgOp>(op));
  std::string name(op->getName().getStringRef().str());
  std::string fun = "";
  for (NamedAttribute kv : op->getAttrs()) {
    if (UnaryFnAttr ufa = llvm::dyn_cast<UnaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(ufa.getValue()).str() + "_";
    } else if (BinaryFnAttr bfa = llvm::dyn_cast<BinaryFnAttr>(kv.getValue())) {
      fun = stringifyEnum(bfa.getValue()).str() + "_";
    }
  }
  name.reserve(128);
  std::replace(name.begin(), name.end(), '.', '_');
  llvm::raw_string_ostream ss(name);
  ss << "_" << fun;
  for (Type t : op->getOperandTypes()) {
    if (failed(appendMangledType(ss, t)))
      return std::string();
    ss << "_";
  }
  std::string res = ss.str();
  res.pop_back();
  return res;
}
```

[Call native c function from MLIR](https://gist.github.com/dmitriykovalev/c9100bd12a986b50bb404cd1086814d6)

[Polygeist Issue](https://github.com/llvm/Polygeist/issues/285)

[Get symbol name](https://github.com/llvm/Polygeist/issues/235)