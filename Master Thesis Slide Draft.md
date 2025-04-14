
# ONNX Model Inference Using Quantized Posit Format in MLIR Framework

## What is Posit Arithmetic?

- A **floating-point alternative** proposed by John L. Gustafson in 2017.
- Designed to **improve precision, dynamic range, and efficiency** over IEEE 754 floating-point.
## Features

- **Tapered precision**: Higher precision near exponent values close to 0.
- **Wider dynamic range** than same-bit-width IEEE floats.
- **No special cases** like NaN, ±Inf (except a single NaR: “Not a Real”).
- **Compact representation** with better accuracy per bit.

(Insert dynamic range image)

---

## What is ONNX (Open Neural Network Exchange)

- ONNX is an open format for representing machine learning models.
- Developed by **Microsoft and Facebook**
- Enabling **interoperability** across different deep learning frameworks like PyTorch, TensorFlow, and tools like OpenVINO, TensorRT.

---
## What is MLIR

- **MLIR (Multi-Level Intermediate Representation)** is a compiler infrastructure project under the LLVM umbrella.
- Designed to **unify and optimize** code across **multiple abstraction levels**.
## Features:

- **Extensibility**: Easy to build custom dialects and passes.
- **Unification**: Bridges the gap between different levels of IR (high-level → hardware).
- **Optimization-friendly**: Reuses LLVM’s robust infrastructure with added tensor semantics.
- **Framework Integration**: Backbone of modern compilers like TensorFlow XLA, Torch-MLIR, IREE.

---
## Motivation:

- Posit arithmetic enables better

