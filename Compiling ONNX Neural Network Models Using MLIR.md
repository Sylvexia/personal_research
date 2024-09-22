Author: Tian Jin, Gheorghe-Teodor Bercea, Tung D. Le, Tong Chen, Gong Su, Haruki Imai, Yasushi Negishi, Anh Leu, Kevin O’Brien, Kiyokuni Kawachiya, Alexandre E. Eichenberger
## What Does this project do?

Compiles the trained ONNX model with mlir, which has minimum runtime support.

## What does the project provide?

-  ONNX Dialect that can be integrated in other projects,
-  compiler interfaces that lower ONNX graphs into MLIR files/LLVM bytecodes/C & Java libraries,
-  `onnx-mlir` driver to perform these lowering,
-  Python/C/C++/Java runtime environment.
## Motivation:

- Training and inference are often done on different environments due to their different optimization characteristics.
	- it is desirable to dynamically rewrite a trained model so that it runs efficiently on a target environment.
- Many deep learning frameworks utilize a highly-optimized library written for a target accelerator.  Rewriting a model for inference consists of replacing the operations in the model with the function calls in the library.
	- Drawbacks
		1. Number of models that can be rewritten is limited by the provided functions in the library.
		2. users need to install additional packages
		3. lacks the ability to tailor code specific to different problems since the same function may be used for them
	- Proposed Solution: developing a compiler that rewrites a trained model to native code for a target hardware
## What is ONNX (Open Neural Network Exchange)?

- Open source format for exchanging AI models.
- Protocol Buffers definition language.
- Defines an extensible computational graph model, operators, and standard data types, which provides a common IR for different frameworks.
![[Pasted image 20231220142407.png]]
![[Pasted image 20231220142453.png]]
```
ir_version: 8
producer_name: "pytorch"
producer_version: "2.1.1"
graph {
  node {
    input: "input.1"
    input: "onnx::Conv_1396"
    input: "onnx::Conv_1397"
    output: "/features/features.0/features.0.0/Conv_output_0"
    name: "/features/features.0/features.0.0/Conv"
    op_type: "Conv"
    attribute {
      name: "dilations"
      ints: 1
      ints: 1
      type: INTS
    }
    attribute {
      name: "group"
      i: 1
      type: INT
    }
    attribute {
      name: "kernel_shape"
      ints: 3
      ints: 3
      type: INTS
    }
    attribute {
      name: "pads"
      ints: 1
      ints: 1
      ints: 1
      ints: 1
      type: INTS
    }
    attribute {
      name: "strides"
      ints: 2
      ints: 2
      type: INTS
    }
}
```
## What is MLIR?

- MLIR is a three-address static single assignment (SSA)-based IR

	- Static single assignment form (SSA) 
		- Requires each variable to be assigned exactly once and defined before it is used
			- Existing variables in the original IR are split into versions
			- Every definition gets its own version.
		- For example:
```
y := 1
y := 2
x := y
```
```
y1 := 1
y2 := 2
x1 := y2
```

- Three-address code: ```X = Y op Z
- In MLIR, Tensor and MemRef types are syntactically represented as tensor〈D 1 ×D 2 × . . . ×D N ×dtype〉 and memref<D 1 ×D 2 × . . . ×D N ×dtype>
- Operation: 
	- TableGen-based
### Compiling ONNX Models

#### Overview
- 5 main dialects:
	- onnx
		- Represents an ONNX model in MLIR language
		- Onnx -> tablegen-based operation definitions
		- Represent all necessary information, such as inputs, outputs, attributes, and description to human-redable.
	- krnl:
		- computation kernel
		- aims to host both loop optimization and scalar semantic optimization in a single representation.
	- affine: 
		- affine transformation
		- Polyhedral structures
		- mainly for dependance analysis and loop transformation
	- std: standard operations such as load, store, addi, addf, absf, and call.
	- llvm: 
		- Wrapping LLVM IR types and instructions into MLIR types and operations
- 4 abstraction layers:
	- 1st (ONNX): High-level representation of ONNX operations.
	- 2nd (Krnl): intermediate dialect for lowering the onnx dialect into low-level dialects
	- 3rd (affine, std): Applying existing optimization passes in MLIR.
	- 4th (LLVM): LLVM for generate bitcode
![[Pasted image 20231219060100.png]]
#### ONNX dialect
```
# src/Dialect/ONNX/ONNXOPs.td.inc
def ONNXLeakyReluOp:ONNX_Op<"LeakyRelu",
  [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ONNX LeakyRelu operation";
  let description = [{
  LeakyRelu takes input data (Tensor<T>) and an argument alpha, and produces one
  output data (Tensor<T>) where the function `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`, is applied to the data tensor elementwise.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$X,
    DefaultValuedAttr<F32Attr, "0.01">:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[BF16]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>]>:$Y);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return 1;
    }
    static int getNumberOfResults() {
      return 1;
    }
    static std::vector<int> getTypeMap() {
      return {30};
    }
  }];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, llvm::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNXLeakyReluOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}
```
