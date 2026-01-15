# Flux Language Specification

Flux is an indentation-based programming language optimized for AI/ML orchestration.

## Variables

Variables are declared using the `let` keyword. Flux supports **Gradual Typing**, meaning type annotations are optional.

```python
# Untyped (any)
let x = 10

# Explicitly typed
let y: int = 20
let pi: float = 3.14
let name: string = "Flux"
let is_valid: bool = True
```

### Assignment
Variables can be reassigned (currently Flux allows overriding names in the same scope, leaning towards a simple environment model).

## Expressions

Flux supports standard algebraic and logical expressions.

### Arithmetic Operators
- `+`, `-`, `*`, `/`, `%`
- `**` (Power)
- `@` (Matrix Multiplication - for Tensors)

### Comparison & Logical Operators
- `==`, `!=`, `<`, `>`, `<=`, `>=`
- `&&` (And), `||` (Or), `!` (Not)
- `in`, `not in` (for Lists/Strings)

### Tensor Expressions
Tensors support element-wise operations with other tensors or scalars.
```python
let t1 = ones([2, 2])
let t2 = t1 * 2.0  # Scalar multiplication
let t3 = t1 + t2   # Element-wise addition
```

## Evaluation Rules

Flux is an interpreted language with the following evaluation characteristics:

1.  **Strict Evaluation**: Arguments to functions are evaluated before the function is called.
2.  **Environment-based Scoping**: Variables are stored in a hierarchical environment. Functions capture their lexical environment (closures).
3.  **Indentation-based Blocks**: Blocks of code (in `if`, `while`, `def`, etc.) are defined by their indentation level.
4.  **Automatic Type Coercion**: In arithmetic operations involving `int` and `float`, `int` is automatically promoted to `float`.
5.  **Static Type Checking Pass**: If type hints are present, a static analysis pass runs before execution to ensure consistency.

## Functions

Functions are defined using the `def` keyword.

```python
def multiply(a: float, b: float) -> float:
    return a * b
```
If no return type is specified, it defaults to `any`.
