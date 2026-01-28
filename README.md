# Flux Language

Flux is a high-performance, AI-native programming language designed to bridge the gap between Python's developer productivity and Rust's runtime efficiency. 

> **Performance Goal**: Minimize Python interpreter overhead in AI orchestration by providing a lean, Rust-powered execution environment for Tensor workloads.

---

## Design Philosophy & Specifications

### 1. High-Speed Orchestration
Frameworks like PyTorch are powerful but often bottlenecked by the Python Global Interpreter Lock (GIL) and VM overhead during complex coordination of small kernels. Flux eliminates this by keeping the entire execution loop inside a memory-safe Rust binary.

### 2. Gradual Typing System
Flux offers a unique blend of dynamic flexibility and static safety.
- **Dynamic by Default**: Move fast without boilerplate.
- **Static when Needed**: Use type hints for critical path validation and documentation.

### 3. AI-First Standard Library
Unlike general-purpose languages, Flux treats `Tensor` as a first-class citizen, with native syntax and optimized operators.

---

## Flux vs. Mojo

While Mojo is a powerful achievement in the AI infrastructure space, Flux carves out a different niche:

- **Rust Ecosystem Affinity**: Flux is built from the ground up to integrate seamlessly with the Rust ecosystem, leveraging its safety guarantees without the complexity of a manual memory management model for the end-user.
- **Lightweight Runtime**: Flux focuses on a minimal runtime footprint, making it ideal for lean AI orchestration where Python's VM overhead is a bottleneck.
- **Design Philosophy**: Flux prioritizes a "Python-like simplicity, Rust-like reliability" approach, specifically optimized for high-speed coordination of Tensor workloads.

---

## Language Guide

### Syntax Overview
Flux uses a clean, indentation-based syntax similar to Python.

```python
# Function with type hints
def compute_loss(pred: tensor, target: tensor) -> float:
    let diff = pred - target
    return mean(diff * diff)

let p = rand([4, 4])
let t = ones([4, 4])
print("Loss:", compute_loss(p, t))
```

### Type System
The static type checker runs before execution, catching errors early.

| Type | Description |
| :--- | :--- |
| `int` | 64-bit signed integer |
| `float` | 64-bit IEEE floating point |
| `string` | UTF-8 encoded string |
| `bool` | Boolean (`True`, `False`) |
| `list[T]` | Generic list (e.g., `list[int]`) |
| `tensor` | Multi-dimensional array (f64) |
| `any` | Opt-out of type checking |

---

## Advanced Features
- **Comprehensions**: Native support for list, set, and dictionary comprehensions.
- **Triple-Quote Strings**: Multi-line literals using `"""` or `'''`.
- **Set Support**: Native set literals and operations (`|`, `&`, etc.).
- **Utility Functions**: Enhanced standard library with `copy()`, `isalnum()`, `islower()`, `isupper()`, `split()`, `round()`, etc.

---

## Python Interoperability
Leverage the massive Python ecosystem directly from Flux.

```python
import numpy as np

let arr = np.array([1, 2, 3])
print("Numpy Array from Flux:", arr)
```
- **Bi-directional conversion**: Pass Tensors between Flux and NumPy seamlessly.
- **Exception Handling**: Python errors are translated into Flux errors with accurate source mapping.

### Direct Module Import (`py_import`)

For cases where you need more fine-grained control or specific Python libraries:

```python
# Import powerhouses
let np = py_import("numpy")
let torch = py_import("torch")

# Use them just like in Python
let arr = np.array([1, 2, 3])
let t = torch.tensor([4.0, 5.0, 6.0])

print("Numpy:", arr)
print("PyTorch:", t)
```

---

## Getting Started

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (stable)

### Installation
```bash
git clone https://github.com/cmc-labo/flux.git
cd flux
cargo build --release
```

### Usage
- **Run Script**: `cargo run -- path/to/script.fl`
- **Interactive REPL**: `cargo run` (Start typing Flux code directly!)

---

## Roadmap & Support Matrix

| Feature | Status | Technology |
| :--- | :--- | :--- |
| **Basic Control Flow** | Full | If, While, For-In, Lambda |
| **Data Structures** | Robust | List, Dict, Set |
| **Strings** | Robust | F-Strings, Slicing |
| **Static Type Hints** | Initial | `TypeChecker` pass |
| **Tensor Ops** | CPU | `ndarray` backend |
| **Python Interop** | Robust | `PyO3` |
| **GPU Support** | Planned | `wgpu` or `Burn` |
| **AOT Compiler** | Backlog | LLVM/MLIR |

---

## License
MIT

---

**Note**: This project is unrelated to the Flux image model (Black Forest Labs) or the InfluxData query language.
