洪祐鈞

# TODO

- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- yolov4 fptosi, math.floor, math.log
- ssd-10 failed: arith.maxnumf, arith.minnumf
- multiple output would failed

# Operations need to support

# LMs

- BERT-Squad: This model answers questions based on the context of the given input paragraph.
- BiDAF: This model is a neural network for answering a query about a given context paragraph.
	- Smallest but cannot run
- GPT-2: Transformer-based language model for text generation.
- GPT-2 with Beam Search Generation: Transformer-based language model for text generation.
- RoBERTa: Transformer-based language model for text generation.
- T5: provide great flexibility and provide better semantic understanding through the training of multiple tasks at once.

# Visions

- version-RFB-320
	- multiple output would failed

# Supported operation at positwrapper

basic, compare, select, fptosi, sitofp, maxnum, minnum
abs, sqrt, rsqrt, exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, erf, log, floor, ceil, trunc, round, max, min, neg

  populateReturnPositOpPatterns(arith::AddFOp{}, "add");
  populateReturnPositOpPatterns(arith::SubFOp{}, "sub");
  populateReturnPositOpPatterns(arith::MulFOp{}, "mul");
  populateReturnPositOpPatterns(arith::DivFOp{}, "div");
  populateReturnPositOpPatterns(arith::NegFOp{}, "neg");
  populateReturnPositOpPatterns(arith::MaxNumFOp{}, "maxnum");
  populateReturnPositOpPatterns(arith::MinNumFOp{}, "minnum");
  populateReturnPositOpPatterns(arith::SelectOp{}, "select");
  populateReturnPositOpPatterns(arith::MaximumFOp{}, "max");
  populateReturnPositOpPatterns(arith::MinimumFOp{}, "min");
  populateReturnPositOpPatterns(math::AbsFOp{}, "abs");
  populateReturnPositOpPatterns(math::SqrtOp{}, "sqrt");
  populateReturnPositOpPatterns(math::RsqrtOp{}, "rsqrt");
  populateReturnPositOpPatterns(math::ExpOp{}, "exp");
  populateReturnPositOpPatterns(math::SinOp{}, "sin");
  populateReturnPositOpPatterns(math::CosOp{}, "cos");
  populateReturnPositOpPatterns(math::TanOp{}, "tan");
  populateReturnPositOpPatterns(math::AsinOp{}, "asin");
  populateReturnPositOpPatterns(math::AcosOp{}, "acos");
  populateReturnPositOpPatterns(math::AtanOp{}, "atan");
  populateReturnPositOpPatterns(math::SinhOp{}, "sinh");
  populateReturnPositOpPatterns(math::CoshOp{}, "cosh");
  populateReturnPositOpPatterns(math::TanhOp{}, "tanh");
  populateReturnPositOpPatterns(math::ErfOp{}, "erf");
  populateReturnPositOpPatterns(math::LogOp{}, "log");
  populateReturnPositOpPatterns(math::FloorOp{}, "floor");
  populateReturnPositOpPatterns(math::CeilOp{}, "ceil");
  populateReturnPositOpPatterns(math::TruncOp{}, "trunc");
  populateReturnPositOpPatterns(math::RoundOp{}, "round");

# Operation need to lower

skip remf uitofp
a{tan}2