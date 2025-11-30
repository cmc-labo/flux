# Flux Language

Flux is an experimental programming language designed to combine the simplicity of Python with the performance and safety of Rust, specifically targeting AI/ML workloads.

## Features

- **Python-like Syntax**: Clean, indentation-based code structure.
- **Rust-Powered**: Built in Rust for memory safety and performance.
- **AI/ML Native**: First-class support for Tensor operations (Work in Progress).

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version)

### Installation

Clone the repository:

```bash
git clone https://github.com/cmc-labo/flux.git
cd flux
```

### Usage

Run a script using the interpreter:

```bash
cargo run -- path/to/script.fl
```

### Example

Create a file named `example.fl`:

```python
def add_tensors(size):
    let t1 = tensor(size)
    let t2 = tensor(size)
    return t1 + t2

let result = add_tensors(3)
print(result)
```

Run it:

```bash
cargo run -- example.fl
```

## Roadmap

- [x] Basic Lexer & Parser (Indentation support)
- [x] Interpreter with Variables & Functions
- [x] Basic Tensor Type & Addition
- [ ] Matrix Multiplication
- [ ] GPU Acceleration (wgpu/CUDA)
- [ ] Python Interop

## License

MIT
