
# Summary

- Try to modify the ONNX model, half way through?

# The current way:

1. Load the ONNX model
2. Build a consumer map for traversal and locate the target deletion node
3. Find all downstream nodes (BFS) and remove them
4. Append last remaining node’s output as graph output
	1. The shape is odd, thinking a way how to get the dimension.
5. Save the ONNX Model

# Not deleted

![](Pasted%20image%2020250522053448.png)

# Deleted

![](Pasted%20image%2020250522053301.png)

# Future Works

- FP16 results.
- Master Thesis.