# TODO

- Move pass after `cf` dialect.
- Math Dialect lowering.
- How does runtime work?
	- Test pipeline.
- Posit Dialect.
# Summary

# Runtime

[pipeline link](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/Model_20Zoo_20Report/)

The output from the link is run through RunONNXModel.py

which is called by RunONNXModelZoo.py

## RunONNXModelZoo.py

`ok, msg = execute_commands(RUN_ONNX_MODEL_CMD + options, tmout=1800)`

options = compile_args + data_set + model

## RunONNXModel.py

clone github and download dataset and model.
# Metric of GPT-2

- Perplexity
	- how well a probability model predicts a sample and is commonly used to evaluate language models, lower the better.
- BLEU Score
	- Measures the similarity between the generated text and reference text, often used in machine translation
- ROUGE Score
	- Evaluates the overlap between the generated text and reference summaries, useful for summarization tasks