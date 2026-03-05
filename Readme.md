#  ![alt text](assets/image.png) 
# Backpropagator

**Backpropagator** project is a project made by me and my friend for understanding how backpropagation works![alt text](image.png) by building a computational graph and automating the chain rule from the ground up.



## 📖 About
In this project, my friend and I implemented **Backpropagation** using only Python and the `math` library. While modern frameworks like PyTorch handle massive tensors on GPUs, **Backpropagator** works at the scalar level, making it the perfect educational tool to understand exactly how gradients flow through a neural network.

## ✨ Key Features
* **Scalar-Level Autograd**: A custom `Value` class that stores data and its derivative (gradient).
* **Automated Chain Rule**: Uses a topological sort to ensure gradients are propagated in the correct order.
* **Full Neural Stack**: Includes building blocks for `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron) architectures.
* **Mathematical Operations**: Supports `+`, `-`, `*`, `/`, `**`, and activation functions like `tanh` and `exp`.
* **PyTorch Verified**: Includes benchmarks to verify our gradient calculations against industry-standard results.

## Class Diagram
#  ![alt text](assets/class-diagram.png) 