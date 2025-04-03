洪祐鈞

# Summary

- In MNIST model,  it still have NaR result in (8, 0) (8,1)
	- Even if there's only arith operation
		- arith.addf
		- arith.addi
		- arith.cmpf
		- arith.cmpi
		- arith.constant
		- arith.maxsi
		- arith.minsi
		- arith.mulf
		- arith.muli
		- arith.select
		- arith.subi
		- arith.xori
	- No math operation like tanh() or divide
	- Divide to zero case can be not consider first.
	- Currently we reduced our problem to the following:
		- 1. Potential error in fp32 conversion
			- Currently the conversion implementation is hardcode by me, might need to migrate to universal.
		- 2. Edge case in universal add, multiplication
- Spend too much time try to write operation dumper
	- Our main compiler block stdout and stderr.
		- onnx-mlir: block stdout and stderr
		- onnx-mlir-opt: can make custom log but you need to get all the pass pipeline, which is hard when the case is taking onnx as input.
	- How we reduce the problem is based on MLIR debug log, and I'm not able to add my own logger.