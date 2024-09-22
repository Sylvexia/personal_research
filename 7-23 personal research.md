## Tips:

1. Find ```llvm::cl``` for compiler option, for onnx-mlir project it;s locate at ```src/Compiler/CompilerOptions.cpp```
2. main() -> compileModule -> addPasses -> determineInputIRLevel -> addCompilerPasses
	1. addONNXToMLIRPasses
	2. addONNXToKrnlPasses
	3. addKrnlToAffinePasses
	4. addKrnlToLLVMPasses
3. addONNXToKrnlPasses -> pm.addPass(onnx_mlir::createLowerToKrnlPass) -> make_unique<FrontendToKrnlLoweringPass> -> FrontendToKrnlLoweringPass::runOnOperation() -> target.addLegalDialect<KrnlDialect> target.addLegalOp<::mlir::UnrealizedConversionCastOp>() -> populateONNXToKrnlConversionPattern() -> populateLoweringONNXConvOpPattern() -> applyPartialConversion()
4. populateLoweringONNXConvOpPattern
	1. patterns.insert<ONNXConvOpToCall>(typeConverter, ctx, opsForCall)
		1. match
		2. rewrite
	2. patterns.insert<ONNXConvOpLowering>(typeConverter, ctx, enableParallel);
		1. matchAndRewrite

[rewrite](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)