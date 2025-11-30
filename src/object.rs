use crate::ast::Block;
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use crate::environment::Environment;

#[derive(Debug, PartialEq, Clone)]
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
        }
    }
}
