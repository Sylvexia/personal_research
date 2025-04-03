洪祐鈞

# Summary

- In MNIST model,  it still have NaR result in (8, 0) (8,1)
	- Even if there's only arith operation
		- arith.addf
		- arith.addi
		- arith.cmpf
		- arith.cmpi
		- arith.constant
arith.maxsi
arith.minsi
arith.mulf
arith.muli
arith.overflow
arith.select
arith.subi
arith.xori
	- No math operation like 
	- Currently we reduced our problem to the following:
		- 1. Potential error in 
		- 