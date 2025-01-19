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

# Operation need to lower

skip remf uitofp
- arith
	- fptosi
	- maximumf
	- maxnumf
	- minimumf
	- minnumf
	- negf
- math
	- floor
	- log

3 
1. 2 seconds periods, wait for response, 5 minutes per conversation
2. Respond other people respond, no other conversation
3. remember to take the notes

advice
- 