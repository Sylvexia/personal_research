
## What is Posit Arithmetic?

- Invented by

---

What is Posit Arithmetic?

Posit arithmetic is a number representation system introduced by John L. Gustafson in 2017, designed to offer higher accuracy and larger dynamic range than floating-point numbers while using fewer bits. It features a tapered precision, allocating more bits to numbers around zero, where most deep learning computations occur, and simplifying exception handling. This makes it particularly appealing for deep learning inference, which often involves matrix multiplications and activations requiring high precision but also efficiency.

Key properties include:

- **Higher Accuracy**: Posits favor decimal accuracy in the central range, reducing representation errors.
- **Efficiency**: Simplified exception handling reduces computational overhead, potentially lowering energy consumption.
- **Dynamic Range**: Offers a larger range than floating-point for the same bit width, crucial for representing diverse model weights.
- **Reproducibility**: Guarantees bitwise-identical results, enhancing reproducibility in scientific computing.