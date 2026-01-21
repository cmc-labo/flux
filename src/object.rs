use crate::ast::Block;

use std::fmt;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::hash::{Hash, Hasher}; // Import Hasher trait
use pyo3::prelude::*;
use crate::environment::Environment;
use crate::tensor::Tensor;

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
    List(Rc<RefCell<Vec<Object>>>),
    Dictionary(Rc<RefCell<HashMap<Object, Object>>>),
    Slice { start: Option<i64>, stop: Option<i64>, step: i64 },
    Tensor(Tensor),
    PyObject(Py<PyAny>),
    Module { name: String, env: Rc<RefCell<Environment>> },
    Break,
    Continue,
}

impl Clone for Object {
    fn clone(&self) -> Self {
        match self {
            Object::Integer(i) => Object::Integer(*i),
            Object::Float(f) => Object::Float(*f),
            Object::String(s) => Object::String(s.clone()),
            Object::Boolean(b) => Object::Boolean(*b),
            Object::Null => Object::Null,
            Object::ReturnValue(v) => Object::ReturnValue(v.clone()),
            Object::Function { params, body, env } => Object::Function { 
                params: params.clone(), 
                body: body.clone(), 
                env: env.clone() 
            },
            Object::NativeFn(f) => Object::NativeFn(*f),
            Object::List(l) => Object::List(l.clone()),
            Object::Dictionary(d) => Object::Dictionary(d.clone()),
            Object::Slice { start, stop, step } => Object::Slice { 
                start: *start, 
                stop: *stop, 
                step: *step 
            },
            Object::Tensor(t) => Object::Tensor(t.clone()),
            Object::PyObject(p) => {
                pyo3::Python::attach(|py| {
                    Object::PyObject(p.clone_ref(py))
                })
            },
            Object::Module { name, env } => Object::Module { 
                name: name.clone(), 
                env: env.clone() 
            },
            Object::Break => Object::Break,
            Object::Continue => Object::Continue,
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
            (Object::List(l), Object::List(r)) => *l.borrow() == *r.borrow(), 
            (Object::Dictionary(l), Object::Dictionary(r)) => {
                let l_map = l.borrow();
                let r_map = r.borrow();
                if l_map.len() != r_map.len() {
                    return false;
                }
                for (k, v) in l_map.iter() {
                    match r_map.get(k) {
                        Some(rv) => if v != rv { return false; },
                        None => return false,
                    }
                }
                true
            },
            (Object::Slice { start: s1, stop: e1, step: st1 }, Object::Slice { start: s2, stop: e2, step: st2 }) => s1 == s2 && e1 == e2 && st1 == st2,
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
                pyo3::Python::attach(|py| {
                    let s = val.bind(py).repr()
                        .map(|r| r.to_string())
                        .unwrap_or_else(|_| "PyObject(error)".to_string());
                    write!(f, "{}", s)
                })
            },
            Object::List(l) => {
                let list = l.borrow();
                let elements: Vec<String> = list.iter().map(|e| e.to_string()).collect();
                write!(f, "[{}]", elements.join(", "))
            },
            Object::Dictionary(d) => {
                let dict = d.borrow();
                let elements: Vec<String> = dict.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
                write!(f, "{{{}}}", elements.join(", "))
            },
            Object::Break => write!(f, "break"),
            Object::Continue => write!(f, "continue"),
            Object::Module { name, .. } => write!(f, "module({})", name),
            Object::Slice { start, stop, step } => write!(f, "slice({:?}, {:?}, {})", start, stop, step),
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
            Object::List(l) => !l.borrow().is_empty(),
            Object::Dictionary(d) => !d.borrow().is_empty(),
            _ => true,
        }
    }
}
