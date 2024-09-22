Robin, Hung
## What might be the future plan?

### Recall and Acknowledge:

- Various research paper shows that posit<8,0> is the best for quantize the model from fp32, which should be supported first.
- Recall what we did in [onnx-mlir](https://github.com/onnx/onnx-mlir). The whole end-to-end process would be:
	1. We input onnx model to onnx-mlir tool.
	2. Compile the model to model.mlir and to model.so
	3. Combine the runtime and model.so, we write a driver program to it
		- [How the driver code looks like?](https://github.com/onnx/onnx-mlir/blob/225c8c4c034cd9c21573218437667121314353de/docs/mnist_example/mnist-runPyRuntime.py#L4)
	4. Given input data and get the inference output.
- There are 4 dialect need to be acknowledged:
	1. onnx dialect: Intuitively, it just represent how onnx operator would look like with mlir perspective.
	2. krnl dialect: Hosting both loop optimization and scalar semantic optimization in a single representation. Intuitively, it just decompose the onnx operator to loop transformation and can be treat as a glue dialect to bridge to existing affine dialect.
	3. affine dialect: Existing dialect for loop representation in mlir, existing optimization can be applied here.
	4. llvm dialect: Wrapping the LLVM IR types and instructions into MLIR types and operations.

### The most proper way:

1. Add posit type to the onnx project, and potentially add to llvm/mlir project so that the following tablegen for example can accept posit data type:
	```cpp
	// ONNXOps.td
	def ONNXAddOp:ONNX_Op<"Add",

		[Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, 
			DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {

		let arguments = (ins AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, 
			TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, 
			TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, 
			TensorOf<[F64]>, TensorOf<[BF16]>]>:$A,

		AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, TensorOf<[UI32]>, 
			TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, TensorOf<[I32]>,
			TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, 
			TensorOf<[BF16]>]>:$B);

		let results = (outs AnyTypeOf<[TensorOf<[UI8]>, TensorOf<[UI16]>, 
			TensorOf<[UI32]>, TensorOf<[UI64]>, TensorOf<[I8]>, TensorOf<[I16]>, 
			TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[F16]>, TensorOf<[F32]>, 
			TensorOf<[F64]>, TensorOf<[BF16]>]>:$C);
	...
	```
2.  When onnx-mlir load .onnx model. The posit type propagate from frontend to onnx dialect
	
	For ONNX dialect side, instead of:
	```cpp
	func.func private @test_add(%arg0 : tensor<f32>, %arg1 : tensor<f32>) 
		-> tensor<f32> {
		
		%0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
	
		"func.return"(%0) : (tensor<f32>) -> ()
	}
	```

	We're more likely to see:
	
	```cpp
	func.func private @test_add(%arg0 : tensor<posit8>, %arg1 : tensor<posit8>) 
		-> tensor<posit8> {
		
		%0 = "onnx.Add"(%arg0, %arg1) : (tensor<posit8>, tensor<posit8>) -> 
			tensor<posit8>
	
		"func.return"(%0) : (tensor<posit8>) -> ()
	}
	```
3.  When onnx dialect converted to krnl dialect, Instead of:
	```cpp
	module {
		func.func private @test_add(%arg0: memref<f32>, %arg1: memref<f32>) -> 
			memref<f32> {
		    %c1 = arith.constant 1 : index
		    %c1_0 = arith.constant 1 : index
		    %alloc = memref.alloc() : memref<f32>
		    %0 = krnl.load %arg0[] : memref<f32>
		    %1 = krnl.load %arg1[] : memref<f32>
		    %2 = arith.addf %0, %1 : f32
		    krnl.store %2, %alloc[] : memref<f32>
		    return %alloc : memref<f32>
		}
	}
	```

	we want this:
	```cpp
	module {
		func.func private @test_add(%arg0: memref<posit8>, %arg1: 
			memref<posit8>) -> memref<posit8> {
		    %c1 = arith.constant 1 : index
		    %c1_0 = arith.constant 1 : index
		    %alloc = memref.alloc() : memref<posit8>
		    %0 = krnl.load %arg0[] : memref<posit8>
		    %1 = krnl.load %arg1[] : memref<posit8>
		    // notice instead of arith dialect, we now need a posit dialect to support
		    %2 = posit8.addf %0, %1 : posit8
		    krnl.store %2, %alloc[] : memref<posit8>
		    return %alloc : memref<posit8>
		}
	}
	```
4. We now add a pass to convert from posit dialect to arith equivalent or function call.
	Or for the example above, convert `%2 = posit.addf %0, %1 : posit8` to the following:
	1. posit arith equivalent: represent the [softposit implementation](https://gitlab.com/cerlane/SoftPosit/-/blob/master/source/s_addMagsP16.c) in mlir arith dialect:
	2. [function call mapping](https://discourse.llvm.org/t/mlir-how-do-i-link-an-external-c-function-for-an-operation-in-an-mlir-file/1821/4)to posit library call.
	The first approach might me the most correct way to do, since later the instruction can be optimize by the existing pass later. However it's a trouble to get it right. 
	Second approach might be a short term fix, but the performance will be very slow since when function call, mlir won't know about softposit mlir instruction would be like, and there's no way of inline the call to it.
5. If done correctly, they all should be able to converted to llvm dialect.
6. Link the compiled model.so and runtime, and user write the model driver code.
7. Add support for the inference driver code. The driver code can be reference [here](https://github.com/onnx/onnx-mlir/blob/f11a21c6cb3777e435beb596744363655b0da9ae/docs/mnist_example/mnist-runPyRuntime.py)

### Potentially not proper way:

The following plan was recall and amalgamation of the previous plans, although there's already potential way of breaking. The though process was worth review for rapid prototype.

1. Does not implement posit dialect or interface. We treat everything as normally we do. We modify the raw data of the onnx model to posit, but the data type is still fp32.
2. After the onnx dialect get converted to krnl dialect, it means that the onnx operators has been transformed to loop representation, so we just capture all `%2 = arith.addf %0, %1 : f32` and all fp32 operation and convert to like what we stated before, no posit dialect needed:
	1. posit arith equivalent
	2. function call mapping
3. Converted to llvm dialect
4. We write a tool to automate the process above, not exposing any user driver code or any posit data type, just input the model, select the model type(like image classification, text classification...), and let it take care of the rest to carry out the output.

There might be some pitfall doing this:

1. You cannot distinguish which part is for weight and which part is just the operator input. There's also multiple pass you may not know about like bufferization, deallocation when converting to llvm dialect. Which may add or delete the instruction you might not know, introducing risk of correctness.
2. As a compiler infrastructure, sharing the posit interface should be done, it should be our one of the research motivation.

## How do I figure out the future plan?

The following is the personal note for how I get the information. It's quite messy and probably only I can read, sorry I ain't got much time
### Log out the conversion:

```bash
./onnx-mlir /home/sylvex/mnist_export/mnist_model.onnx -debug-only=dialect-conversion &> change.txt
```

> open the file with non-gui text editor

![[change.txt]]

take a view on how onnx.Conv get lower to arith, memref, krnl, affine dialect

```bash
//===-------------------------------------------===//
Legalizing operation : 'onnx.Conv'(0x568c1e75e5d0) {
  %18 = "onnx.Conv"(<<UNKNOWN SSA VALUE>>, %5, %3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/conv1/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x28x28xf32>

  * Fold {
  } -> FAILURE : unable to fold

  * Pattern : 'onnx.Conv -> ()' {
  } -> FAILURE : pattern failed to match

  * Pattern : 'onnx.Conv -> ()' {
    ** Insert  : 'arith.constant'(0x568c1e788e20)
    ** Insert  : 'arith.constant'(0x568c1e788ef0)
	// more constant...
    ** Insert  : 'memref.alloc'(0x568c1e784660)
    ** Insert  : 'arith.constant'(0x568c1e784740)
    ** Insert  : 'arith.constant'(0x568c1e784c60)
	// more constant...
    ** Insert  : 'krnl.define_loops'(0x568c1e784fc0)
    ** Insert  : 'krnl.get_induction_var_value'(0x568c1e2e6d10)
    ** Insert  : 'arith.constant'(0x568c1e78af50)
    ** Insert  : 'affine.apply'(0x568c1e823060)
    ** Insert  : 'affine.apply'(0x568c1e426490)
    ** Insert  : 'arith.constant'(0x568c1e78c2a0)
    ** Insert  : 'krnl.define_loops'(0x568c1e78c380)
    ** Insert  : 'arith.constant'(0x568c1e78c440)
    ** Insert  : 'arith.constant'(0x568c1e78c510)
    ** Insert  : 'krnl.get_induction_var_value'(0x568c1e824a00)
    ** Insert  : 'krnl.define_loops'(0x568c1e78cae0)
    ** Insert  : 'arith.constant'(0x568c1e78cba0)
    ** Insert  : 'arith.constant'(0x568c1e849300)
	// more constant...
    ** Insert  : 'affine.apply'(0x568c1e83d7f0)
    ** Insert  : 'arith.constant'(0x568c1e84a6f0)
    ** Insert  : 'affine.max'(0x568c1e84a7c0)
    ** Insert  : 'affine.apply'(0x568c1e84ad30)
    ** Insert  : 'affine.min'(0x568c1e84b2d0)
    ** Insert  : 'arith.constant'(0x568c1e84b3f0)
    ** Insert  : 'arith.constant'(0x568c1e84b4c0)
	// more constant...
    ** Insert  : 'affine.apply'(0x568c1e84c420)
    ** Insert  : 'arith.constant'(0x568c1e84c560)
    ** Insert  : 'affine.max'(0x568c1e84c630)
    ** Insert  : 'affine.apply'(0x568c1e84c740)
    ** Insert  : 'affine.min'(0x568c1e84cd00)
    ** Insert  : 'krnl.get_induction_var_value'(0x568c1e2e6ef0)
    ** Insert  : 'affine.apply'(0x568c1e84d720)
    ** Insert  : 'arith.constant'(0x568c1e84d890)
    ** Insert  : 'affine.apply'(0x568c1e3307b0)
    ** Insert  : 'arith.constant'(0x568c1e84ded0)
    ** Insert  : 'affine.apply'(0x568c1e4e8360)
    ** Insert  : 'krnl.load'(0x568c1e344340)
    ** Insert  : 'krnl.load'(0x568c1e3309b0)
    ** Insert  : 'arith.mulf'(0x568c1e84ef00)
    ** Insert  : 'arith.addf'(0x568c1e84efe0)
    ** Insert  : 'krnl.yield'(0x568c1e84f0b0)
    ** Insert  : 'krnl.iterate'(0x568c1e463290)
    ** Insert  : 'krnl.load'(0x568c1e84f170)
    ** Insert  : 'arith.addf'(0x568c1e84f220)
    ** Insert  : 'krnl.store'(0x568c1e6f5c00)
    ** Insert  : 'krnl.yield'(0x568c1e84f2c0)
    ** Insert  : 'krnl.iterate'(0x568c1e344240)
    ** Insert  : 'krnl.yield'(0x568c1e84f320)
    ** Insert  : 'krnl.iterate'(0x568c1e84fa80)
    ** Replace : 'onnx.Conv'(0x568c1e75e5d0)

    //===-------------------------------------------===//
    Legalizing operation : 'arith.constant'(0x568c1e788e20) {
      %18 = "arith.constant"() <{value = 3 : index}> : () -> index

    } -> SUCCESS : operation marked legal by the target
    //===-------------------------------------------===//

	// for each insert instruction, there's legalization

```

Observation:
- For onnx.Conv, it was parsed from .onnx model. And it would be lower from onnx to krnl and affine dialect (main compiler dialect of the project)
- This lowering is mainly decompose one big operation into loop transformation, memory operation and arithmetic operation. 

### LLVM Debug

Find the "DEBUG_TYPE" in llvm project is useful for certain component debug, it allows the debug-compiled tool extract necessary information. 

reference:
https://github.com/llvm/llvm-project/blob/254e0abf5be2e98cb7f1fa52617b71f4b94b11a4/llvm/include/llvm/Support/Debug.h#L55-L58

For logging the conversion above, if we trace the llvm codebase, we would find the following snippet in mlir/lib/Transforms/Utils/DialectConversion.cpp

```cpp
#define DEBUG_TYPE "dialect-conversion"
```

and if we were to called the insert mlir implementation

Tracing the logger code from llvm codebase:

Insert: 

```cpp
void ConversionPatternRewriterImpl::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
    
    LLVM_DEBUG({
		logger.startLine() << "** Insert : '" << op->getName() << "'(" << op
		<< ")\n";
	});
	
	assert(!wasOpReplaced(op->getParentOp()) &&
		"attempting to insert into a block within a replaced/erased op");
		
    if (!previous.isSet()) {
        // This is a newly created op.
        appendRewrite<CreateOperationRewrite>(op);
		return;
	}

	Operation *prevOp = previous.getPoint() == previous.getBlock()->end()
		? nullptr
		: &*previous.getPoint();

	appendRewrite<MoveOperationRewrite>(op, previous.getBlock(), prevOp);

}
```

### Inspecting code Structure

In onnx-mlir, we probably can refer to existing implementation to do what we need. We now focus on elementwise-binary operation, because operation like convolution is too complicated. And most we did is just overwriting a <= b op c instruction.

The code snippet below is the code path of the ElementwiseBinaryOp for scalar part:

#### Scalar

```cpp
Value alloc = create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims(), alignment);

// b().create<memref::AllocOp>(loc(), type, dynSymbols, alignmentAttr);

Value lhs = create.krnl.load(operands[0]);

Value rhs = create.krnl.load(operands[1]);

// Apply the element-wise function.

Value result = emitScalarOpFor<ElementwiseBinaryOp>(

rewriter, loc, op, outputElementType, {lhs, rhs});

/*
return rewriter.create<ScalarIOp<Op>>(

loc, elementType, scalarOperands, std::nullopt);
*/

result = opFusionHelper.emitFuseOps(result, alloc);

// I don't know how the fuse operation work... bruh

// Store result in the resulting array.

create.krnl.store(result, alloc);
```

input:

```cpp
func.func private @test_add(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> 
	tensor<f32> {
	
	%0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>

	"func.return"(%0) : (tensor<f32>) -> ()
}
```

command: 

```bash
/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl /home/sylvex/onnx-mlir/exp/mlir/scalar.mlir -split-input-file
```

output:

```bash
module {
  func.func private @test_add(%arg0: memref<f32>, %arg1: memref<f32>) -> memref<f32> {
    %c1 = arith.constant 1 : index  // not sure why the arith.constant is created
    %c1_0 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<f32>
    %0 = krnl.load %arg0[] : memref<f32>
    %1 = krnl.load %arg1[] : memref<f32>
    %2 = arith.addf %0, %1 : f32
    krnl.store %2, %alloc[] : memref<f32>
    return %alloc : memref<f32>
  }
}
```

#### Tensor

This is for comparison to the scalar, as we see, we can see it get populate with more loop. The tensor logic is too complicated for us to achieve our goal.

input:

```
func.func private @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {

%0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>

"func.return"(%0) : (tensor<*xf32>) -> ()
}
```

command:

```bash
/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/Math/Elementwise.mlir -split-input-file
```

result:

```
module {
  func.func private @test_add(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c10_0 = arith.constant 10 : index
    %c10_1 = arith.constant 10 : index
    %c10_2 = arith.constant 10 : index
    %c1_3 = arith.constant 1 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x10xf32>
    %0:2 = krnl.define_loops 2
    %c0 = arith.constant 0 : index
    %c10_4 = arith.constant 10 : index
    %c10_5 = arith.constant 10 : index
    krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 10, %0#1 -> %arg3 = 0 to 10){
      %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      %c10_6 = arith.constant 10 : index
      %c10_7 = arith.constant 10 : index
      %2 = krnl.load %arg0[%1#0, %1#1] : memref<10x10xf32>
      %c10_8 = arith.constant 10 : index
      %c10_9 = arith.constant 10 : index
      %3 = krnl.load %arg1[%1#0, %1#1] : memref<10x10xf32>
      %4 = arith.addf %2, %3 : f32
      krnl.store %4, %alloc[%1#0, %1#1] : memref<10x10xf32>
    }
    return %alloc : memref<10x10xf32>
  }
}
```

### ONNX-mlir Test

#### Why do we care about the test?

- We can know about:
	- what feature implemented?
	- what conversion it had been?
	- what dialect it has been?
	- the command flag to achieve certain input or output.
- For our research, we can write the prototype dialect first, and test our methodology is correct or not.

Knowing this powerful concept can reduce my frustration of meaningless tracing code.

THIS SHOULD BE THE START OF THE WHOLE RESEARCH. WHY DO I WASTE LIKE 2 MONTH OF TIME NOT KNOWING THIS...

Seriously, if you see code like various template code:

```cpp
// Recursive class specialized for AffineBuilderKrnlMem refereed to as
// affineKMem.

template <class... Ts>
struct MultiDialectBuilder<AffineBuilderKrnlMem, Ts...>
	: MultiDialectBuilder<Ts...> {
MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
	: MultiDialectBuilder<Ts...>(b, loc), affineKMem(b, loc) {}
MultiDialectBuilder(const DialectBuilder &db)
	: MultiDialectBuilder<Ts...>(db), affineKMem(db) {}
AffineBuilderKrnlMem affineKMem;
};
```

LIKE WHAT THE F...
#### Test

For ONNX-mlir, the test command is as follow, funny enough, the original project doc gave the wrong command. It was make command but another doc state that the project is built from `ninja`

```bash
ninja check-onnx-lit
ninja check-onnx-backend
ninja check-onnx-backend-dynamic
ninja check-onnx-backend-constant
ninja check-onnx-numerical
```

Now we focus on check-onnx-lit test, since we care about how the mlir is processed.

How the test work is basically llvm-lit and FileCheck, these 2 tools.

#### What is llvm-lit?

> **lit** is a portable tool for executing LLVM and Clang style test suites, summarizing their results, and providing indication of failures. **lit** is designed to be a lightweight testing tool with as simple a user interface as possible.

(https://llvm.org/docs/CommandGuide/lit.html)

The following medium article have state how the llvm-lit get set up.
(https://medium.com/@mshockwave/using-llvm-lit-out-of-tree-5cddada85a78)

#### What is FileTest?

Basically it is just an enhanced grep for software testing.

(https://llvm.org/docs/CommandGuide/FileCheck.html)

#### Running the command

Running the first command:
```bash
ninja check-onnx-lit
```

Output:

![[llvm lit result.png]]

The test is intentionally failed for demo purpose.

Essentially what the command does is just run the entire testsuite.

#### How did it get set up

Basically I guess how it works:

1. Load the python config, like file extension or accelerator.
2. Get the environment from compile information. 
3. For each test, generate the terminal command. 
4. Optionally run the command multi-process and collect the return value.
5. Get final result. If wrong the FileCheck will tell you where it goes wrong.

In CMakeLists.txt The related python config file can be found here:

```bash
configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in

${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py

MAIN_CONFIG

${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
```

How the test target

```bash
add_lit_testsuite(check-onnx-lit

"Running the ONNX-MLIR regression tests"

${CMAKE_CURRENT_BINARY_DIR}

DEPENDS

${ONNX_MLIR_TEST_DEPENDS})

set_target_properties(check-onnx-lit PROPERTIES FOLDER "Tests")

  

add_lit_testsuites(ONNX_MLIR

${CMAKE_CURRENT_SOURCE_DIR}

DEPENDS

${ONNX_MLIR_TEST_DEPENDS})
```

#### Structure of a single test.

In test, it would be something like:

```bash
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {

%0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>

"func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (0, d0)>

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (32, d0 + 2)>

// CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (0, d1)>

// CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (32, d1 + 2)>
...
```

When compile the test, the compile script would be locate at build/test
example: 

```bash
set -o pipefail;set -x;{ { set +x; } 2>/dev/null && echo 'RUN: at line 1': '/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir -split-input-file | /home/sylvex/onnx_llvm/llvm-project/build/bin/FileCheck /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir' >&2 && { set -x; } 2>/dev/null && {   /home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir -split-input-file | /home/sylvex/onnx_llvm/llvm-project/build/bin/FileCheck /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir; }; }
```

Running the command above from the information can achieve running single test.

Like:
```bash
/home/sylvex/onnx-mlir/build/Debug/bin/onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir -split-input-file | /home/sylvex/onnx_llvm/llvm-project/build/bin/FileCheck /home/sylvex/onnx-mlir/test/mlir/conversion/onnx_to_krnl/NN/Conv_with_canonicalize.mlir
```