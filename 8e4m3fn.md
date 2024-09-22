
In onnx-mlir, exclude `*hlo*, **/third_party* , *.md`
The file are as follows:

- gen_onnx_mlir.py
	- generate onnx tablegen
	- TODO: see the doc that run this
- ONNXOPs.td.inc
	- auto-generated file
	- Details can be found in docs/ImportONNXDefs.md
	- only limited ops ares supported
		- example: conv op only support f16 f32 f64
- OpHelper.cpp
	- helper function for lowering onnx ops to krnl dialect.
	- `Type convertONNXTypeToMLIRType(builder, onnxType)`
		- `case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN:`
		- `return builder.getFloat8E4M3FNType();`
	- `mlirTypeToOnnxType(elemType)`
		- convert mlir to onnx type
- OMTensor.java
	- java runtime support?
- BType.cpp
	- `Builder b(ctx);`
	- `case BType::FLOAT8E4M3FN : return b.getFloat8E4M3FNType();`
	- Get the type from `mlir/IR/Builders` from the `llvm/mlir` project
	- imply we need to add interface in mlir!
- BType.hpp
	- Enumeration of types
		- `FLOAT8E4M3FN = 17`
	- Templated basic type traits
		- `#define DEFINE_BTypeCppTypeTraits`
	- dispatchByBType
		- idk
- WideNum.cpp
	- interface and return `type.toAPFloat()`
- WideNum.hpp
	- seemed to support `static_cast<Y>(x) == WideNum::from<X>(btype, x).To<Y>(btype)`
- SmallFP.cpp
	- add `template class SmallFPBase<float_8e4m3fn, 8>;`
- SmallFP.hpp
	- We might need to touch LLVM APFLOAT
	```cpp
	class float_8e4m3fn : public detail::SmallFPBase<float_8e4m3fn, 8> {
		using Base = detail::SmallFPBase<float_8e4m3fn, 8>;
	public:
		using Base::Base;
		static const llvm::fltSemantics &semantics() {
			return llvm::APFloat::Float8E4M3FN();
		}
		static constexpr float max = 448.0f;
	};
	```
	```cpp
	template <>
	struct mlir::DenseElementsAttr::
	is_valid_cpp_fp_type<onnx_mlir::float_8e4m3fn> {
		static constexpr bool value = true;
	};
	```
- fp_data.py
	- test fp8 on onnx 
- TestBType.cpp
- TestSmallFP.cpp
	- llvm apfloat included
	- TODO: investigate further
- onnx_constprop.mlir
- fp8.nonraw_data.json
- fp8_raw_data.json