`builtinAttribute.td`
```tablegen
def Builtin_FloatAttr : Builtin_Attr<"Float", "float", [TypedAttrInterface]> {

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        APFloatParameter<"">:$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
      if (type.isF64() || !::llvm::isa<FloatType>(type))
        return $_get(type.getContext(), type, APFloat(value));

      bool unused;
      APFloat val(value);
      val.convert(::llvm::cast<FloatType>(type).getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &unused);
      return $_get(type.getContext(), type, val);
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = APFloat;

    double getValueAsDouble() const;
    static double getValueAsDouble(APFloat val);
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}
```


mlir:
```cpp
AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
```

```cpp
FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), type, value);
}
```

```cpp
AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
  if (type.isF64() || !::llvm::isa<FloatType>(type))
	return $_get(type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(::llvm::cast<FloatType>(type).getFloatSemantics(),
			  APFloat::rmNearestTiesToEven, &unused);
  return $_get(type.getContext(), type, val);
}]>
```

```cpp
FloatAttr FloatAttr::get(Type type, double value) {
  if (type.isF64() || !::llvm::isa<FloatType>(type))
    return Base::get(type.getContext(), type, APFloat(value));

  // This handles, e.g., F16 because there is no APFloat constructor for it.
  bool unused;
  APFloat val(value);
  val.convert(::llvm::cast<FloatType>(type).getFloatSemantics(),
              APFloat::rmNearestTiesToEven, &unused);
  return Base::get(type.getContext(), type, val);
}

```

```cpp
FloatAttr Builder::getF16FloatAttr(float value) {
  return FloatAttr::get(getF16Type(), value);
}
```

# ConstantOps

```
def Arith_ConstantOp : Op<Arith_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>,
     DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs /*SignlessIntegerOrFloatLike*/AnyType:$result);

  let extraClassDeclaration = [{
    static bool isBuildableWith(Attribute value, Type type);
    
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}
```

# AddFOp

```
def Arith_AddFOp : Arith_FloatBinaryOp<"addf", [Commutative]> {
  let hasFolder = 1;
}
```

```
class Arith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    Arith_BinaryOp<mnemonic,
      !listconcat([Pure, DeclareOpInterfaceMethods<ArithFastMathInterface>],
                  traits)>,
    Arguments<(ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<
        Arith_FastMathAttr, "::mlir::arith::FastMathFlags::none">:$fastmath)>,
    Results<(outs FloatLike:$result)> {
  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}
```