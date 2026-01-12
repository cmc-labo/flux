use crate::ast::Block;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use crate::environment::Environment;
use pyo3::types::PyAnyMethods;

#[derive(Debug)]
pub enum Object {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    ReturnValue(Box<Object>),
    Function { params: Vec<String>, body: Block, env: Rc<RefCell<Environment>> },
    NativeFn(fn(Vec<Object>) -> Result<Object, String>),
    Tensor(crate::tensor::Tensor),
    PyObject(pyo3::Py<pyo3::types::PyAny>),
    List(Vec<Object>),
    Dictionary(std::collections::HashMap<Object, Object>),
    Break,
    Continue,
    Module { name: String, env: Rc<RefCell<Environment>> },
}

impl Clone for Object {
    fn clone(&self) -> Self {
        match self {
            Object::Integer(i) => Object::Integer(*i),
            Object::Float(f) => Object::Float(*f),
            Object::String(s) => Object::String(s.clone()),
            Object::Boolean(b) => Object::Boolean(*b),
            Object::Null => Object::Null,
            Object::ReturnValue(val) => Object::ReturnValue(val.clone()),
            Object::Function { params, body, env } => Object::Function { params: params.clone(), body: body.clone(), env: env.clone() },
            Object::NativeFn(f) => Object::NativeFn(*f),
            Object::Tensor(t) => Object::Tensor(t.clone()),
            Object::PyObject(p) => {
                pyo3::Python::with_gil(|py| {
                    Object::PyObject(p.clone_ref(py))
                })
            },
            Object::List(l) => Object::List(l.clone()),
            Object::Dictionary(d) => Object::Dictionary(d.clone()),
            Object::Break => Object::Break,
            Object::Continue => Object::Continue,
            Object::Module { name, env } => Object::Module { name: name.clone(), env: env.clone() },
        }
    }
}

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Object::Integer(l), Object::Integer(r)) => l == r,
            (Object::Float(l), Object::Float(r)) => l == r,
            (Object::String(l), Object::String(r)) => l == r,
            (Object::Boolean(l), Object::Boolean(r)) => l == r,
            (Object::Null, Object::Null) => true,
            (Object::Tensor(l), Object::Tensor(r)) => l == r,
            (Object::List(l), Object::List(r)) => l == r,
            (Object::Dictionary(l), Object::Dictionary(r)) => l == r,
            // For others, return false for now
            _ => false,
        }
    }
}

impl Eq for Object {}

impl std::hash::Hash for Object {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Object::Integer(i) => i.hash(state),
            Object::String(s) => s.hash(state),
            Object::Boolean(b) => b.hash(state),
            Object::Null => (),
            Object::Float(f) => {
                // Warning: hashing floats is dangerous
                f.to_bits().hash(state);
            }
            _ => {
                // Non-hashable types? Using address or just ignoring?
                // Python lists are not hashable.
                // For now, let's just use discriminant.
            }
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Object::Integer(val) => write!(f, "{}", val),
            Object::Float(val) => write!(f, "{}", val),
            Object::String(val) => write!(f, "{}", val),
            Object::Boolean(val) => write!(f, "{}", val),
            Object::Null => write!(f, "null"),
            Object::ReturnValue(val) => write!(f, "{}", val),
            Object::Function { .. } => write!(f, "function"),
            Object::NativeFn(_) => write!(f, "native_function"),
            Object::Tensor(val) => write!(f, "{}", val),
            Object::PyObject(val) => {
                pyo3::Python::with_gil(|py| {
                    let s = val.bind(py).repr()
                        .map(|r| r.to_string())
                        .unwrap_or_else(|_| "PyObject(error)".to_string());
                    write!(f, "{}", s)
                })
            },
            Object::List(elements) => {
                write!(f, "[")?;
                for (i, el) in elements.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", el)?;
                }
                write!(f, "]")
            }
            Object::Dictionary(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Object::Break => write!(f, "break"),
            Object::Continue => write!(f, "continue"),
            Object::Module { name, .. } => write!(f, "module({})", name),
        }
    }
}

impl Object {
    pub fn is_truthy(&self) -> bool {
        match self {
            Object::Null => false,
            Object::Boolean(b) => *b,
            Object::Integer(i) => *i != 0,
            Object::String(s) => !s.is_empty(),
            Object::List(l) => !l.is_empty(),
            Object::Dictionary(d) => !d.is_empty(),
            _ => true,
        }
    }
}
