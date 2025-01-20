洪祐鈞

# TODO

- Come up with better scheme to keep the download file
	- `def check_model(model_path, model_name, compile_args, report_dir):`
- yolov4 fptosi, math.floor, math.log
- ssd-10 failed: arith.maxnumf, arith.minnumf
- multiple output would failed

# Summary

- Still investigating what model can be run... 
	- (Sorry, recently been pretty busy...)
# LMs

- Models:
	- BERT-Squad: This model answers questions based on the context of the given input paragraph.
	- BiDAF: This model is a neural network for answering a query about a given context paragraph.
		- Smallest but cannot run even in fp32 case
	- GPT-2: Transformer-based language model for text generation.
	- GPT-2 with Beam Search Generation: Transformer-based language model for text generation.
	- RoBERTa: Transformer-based language model for text generation.
	- T5: provide great flexibility and provide better semantic understanding through the training of multiple tasks at once.
- Summary: It's unlikely to be able to run any language models, since other models is quite large.
# Visions

- version-RFB-320
	- The result can output from model, but the conversion from raw posit bit failed 
- yolov4
	- Lowering issue, currently investigating...

# Supported operation

- 22+ arith and math lowering on both wrapper and mlir side
	- Not fully tested on both side since there are a lot of operations
- Supported listing: 
	- arith: add, sub, mul, div, cmp, select, fptosi, sitofp, maxnum, minnum, max, min, neg
	- math: abs, sqrt, rsqrt, exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, erf, log, floor, ceil, trunc, round