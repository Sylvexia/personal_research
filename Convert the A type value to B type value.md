
```cpp
// File: ConvertF32ToF16Pass.cpp

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct ConvertF32ToF16Pass : public PassWrapper<ConvertF32ToF16Pass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](Operation *op) {
      // Convert types
      for (auto &result : op->getResults()) {
        if (result.getType().isF32()) {
          result.setType(builder.getF16Type());
        }
      }

      // Convert operands
      for (auto &operand : op->getOpOperands()) {
        if (operand.get().getType().isF32()) {
          operand.set(builder.create<arith::ExtFOp>(op->getLoc(), builder.getF16Type(), operand.get()));
        }
      }
    });
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createConvertF32ToF16Pass() {
  return std::make_unique<ConvertF32ToF16Pass>();
}

static PassRegistration<ConvertF32ToF16Pass> pass("convert-f32-to-f16", "Convert all f32 types and values to f16");
```