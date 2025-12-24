use crate::ast::{Statement, Expression, Block, PrefixOperator, InfixOperator};
use crate::object::Object;
use crate::environment::Environment;
use std::rc::Rc;
use std::cell::RefCell;
use std::cell::RefCell;
use pyo3::prelude::*;


pub struct Interpreter {
    // Global environment? Or passed to eval?
    // Let's keep it simple: eval takes env.
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {}
    }

    pub fn eval(&mut self, node: Statement, env: Rc<RefCell<Environment>>) -> Result<Object, String> {
        match node {
            Statement::Expression(expr) => self.eval_expression(expr, env),
            Statement::Let { name, value } => {
                let val = self.eval_expression(value, env.clone())?;
                env.borrow_mut().set(name, val.clone());
                Ok(val) // Return value of assignment?
            },
            Statement::Return(expr_opt) => {
                let val = match expr_opt {
                    Some(expr) => self.eval_expression(expr, env)?,
                    None => Object::Null,
                };
                Ok(Object::ReturnValue(Box::new(val)))
            },
            Statement::FunctionDef { name, params, body } => {
                let func = Object::Function { params, body, env: env.clone() };
                env.borrow_mut().set(name, func.clone());
                Ok(Object::Null)
            },
            Statement::If { condition, consequence, alternative } => {
                let cond = self.eval_expression(condition, env.clone())?;
                if self.is_truthy(cond) {
                    self.eval_block(consequence, env)
                } else if let Some(alt) = alternative {
                    self.eval_block(alt, env)
                } else {
                    Ok(Object::Null)
                }
            },
            Statement::While { condition, body } => {
                let mut result = Object::Null;
                loop {
                    let cond = self.eval_expression(condition.clone(), env.clone())?;
                    if !self.is_truthy(cond) {
                        break;
                    }
                    result = self.eval_block(body.clone(), env.clone())?;
                    if let Object::ReturnValue(_) = result {
                        return Ok(result);
                    }
                }
                Ok(result)
            },
            Statement::Print(expr) => {
                let val = self.eval_expression(expr, env)?;
                println!("{}", val);
                Ok(Object::Null)
            },
            Statement::For { variable, iterable, body } => {
                let iter_obj = self.eval_expression(iterable, env.clone())?;
                let elements = match iter_obj {
                    Object::List(l) => l,
                    Object::Tensor(_t) => {
                        // For simplicity, iterate over the first dimension if it's a tensor?
                        // Or just error for now.
                        return Err(format!("Tensor iteration not yet implemented"));
                    },
                    _ => return Err(format!("Cannot iterate over {}", iter_obj)),
                };

                let mut result = Object::Null;
                for el in elements {
                    env.borrow_mut().set(variable.clone(), el);
                    result = self.eval_block(body.clone(), env.clone())?;
                    if let Object::ReturnValue(_) = result {
                        return Ok(result);
                    }
                }
                Ok(result)
            }
        }
    }

    pub fn eval_block(&mut self, block: Block, env: Rc<RefCell<Environment>>) -> Result<Object, String> {
        let mut result = Object::Null;
        for stmt in block.statements {
            result = self.eval(stmt, env.clone())?;
            if let Object::ReturnValue(_) = result {
                return Ok(result);
            }
        }
        Ok(result)
    }

    fn eval_expression(&mut self, expr: Expression, env: Rc<RefCell<Environment>>) -> Result<Object, String> {
        match expr {
            Expression::Integer(i) => Ok(Object::Integer(i)),
            Expression::Float(f) => Ok(Object::Float(f)),
            Expression::String(s) => Ok(Object::String(s)),
            Expression::Identifier(name) => {
                match env.borrow().get(&name) {
                    Some(obj) => Ok(obj),
                    None => Err(format!("Identifier not found: {}", name)),
                }
            },
            Expression::Prefix { operator, right } => {
                let right_val = self.eval_expression(*right, env)?;
                self.eval_prefix_expression(operator, right_val)
            },
            Expression::Infix { left, operator, right } => {
                let left_val = self.eval_expression(*left, env.clone())?;
                let right_val = self.eval_expression(*right, env)?;
                self.eval_infix_expression(operator, left_val, right_val)
            },
            Expression::Call { function, arguments } => {
                let func = self.eval_expression(*function, env.clone())?;
                let args = arguments.into_iter()
                    .map(|arg| self.eval_expression(arg, env.clone()))
                    .collect::<Result<Vec<Object>, String>>()?;
                self.apply_function(func, args)
            },
            Expression::Get { object, name } => {
                let obj = self.eval_expression(*object, env)?;
                match obj {
                    Object::PyObject(py_obj) => {
                        pyo3::Python::with_gil(|py| {
                            let getattr = py_obj.getattr(py, name.as_str())
                                .map_err(|e| format!("Python getattr error: {}", e))?;
                            Ok(Object::PyObject(getattr.into()))
                        })
                    },
                    _ => Err(format!("Property access not supported on {}", obj)),
                }
            },
            Expression::List(elements) => {
                let mut objs = Vec::new();
                for el in elements {
                    objs.push(self.eval_expression(el, env.clone())?);
                }
                Ok(Object::List(objs))
            },
            Expression::Index { object, index } => {
                let obj = self.eval_expression(*object, env.clone())?;
                let idx = self.eval_expression(*index, env)?;
                
                match (obj, idx) {
                    (Object::List(l), Object::Integer(i)) => {
                        if i < 0 || i >= l.len() as i64 {
                            return Err(format!("Index out of bounds: {}", i));
                        }
                        Ok(l[i as usize].clone())
                    },
                    (Object::Tensor(_t), Object::Integer(_i)) => {
                        // Simple indexing for tensor?
                        Err(format!("Tensor indexing not yet fully implemented"))
                    },
                    (o, i) => Err(format!("Cannot index {} with {}", o, i)),
                }
            },
            Expression::ListComprehension { element, variable, iterable, condition } => {
                let iter_obj = self.eval_expression(*iterable, env.clone())?;
                let elements = match iter_obj {
                    Object::List(l) => l,
                    _ => return Err(format!("Cannot iterate over {} in list comprehension", iter_obj)),
                };

                let mut result_list = Vec::new();
                for el in elements {
                    env.borrow_mut().set(variable.clone(), el);
                    
                    let should_include = if let Some(cond) = &condition {
                        let res = self.eval_expression((**cond).clone(), env.clone())?;
                        self.is_truthy(res)
                    } else {
                        true
                    };

                    if should_include {
                        let val = self.eval_expression((*element).clone(), env.clone())?;
                        result_list.push(val);
                    }
                }
                Ok(Object::List(result_list))
            }
        }
    }

    fn eval_prefix_expression(&self, operator: PrefixOperator, right: Object) -> Result<Object, String> {
        match operator {
            PrefixOperator::Minus => match right {
                Object::Integer(i) => Ok(Object::Integer(-i)),
                Object::Float(f) => Ok(Object::Float(-f)),
                _ => Err(format!("Unknown operator: -{}", right)),
            },
            PrefixOperator::Not => match right {
                Object::Boolean(b) => Ok(Object::Boolean(!b)),
                Object::Null => Ok(Object::Boolean(true)),
                _ => Ok(Object::Boolean(false)), // Python-like truthiness?
            },
        }
    }

    fn eval_infix_expression(&self, operator: InfixOperator, left: Object, right: Object) -> Result<Object, String> {
        // Handle logical operators first (they work with any type)
        match operator {
            InfixOperator::And => {
                if !self.is_truthy(left.clone()) {
                    return Ok(left);
                }
                return Ok(right);
            },
            InfixOperator::Or => {
                if self.is_truthy(left.clone()) {
                    return Ok(left);
                }
                return Ok(right);
            },
            _ => {}
        }
        
        // Handle type-specific operators
        match (left, right) {
            (Object::Integer(l), Object::Integer(r)) => match operator {
                InfixOperator::Plus => Ok(Object::Integer(l + r)),
                InfixOperator::Minus => Ok(Object::Integer(l - r)),
                InfixOperator::Multiply => Ok(Object::Integer(l * r)),
                InfixOperator::Divide => Ok(Object::Integer(l / r)),
                InfixOperator::Modulo => Ok(Object::Integer(l % r)),
                InfixOperator::Equal => Ok(Object::Boolean(l == r)),
                InfixOperator::NotEqual => Ok(Object::Boolean(l != r)),
                InfixOperator::LessThan => Ok(Object::Boolean(l < r)),
                InfixOperator::GreaterThan => Ok(Object::Boolean(l > r)),
                InfixOperator::LessThanOrEqual => Ok(Object::Boolean(l <= r)),
                InfixOperator::GreaterThanOrEqual => Ok(Object::Boolean(l >= r)),
                _ => Err(format!("Unknown operator for integers")),
            },
            (Object::Float(l), Object::Float(r)) => match operator {
                 InfixOperator::Plus => Ok(Object::Float(l + r)),
                 // ... simplify
                 InfixOperator::Equal => Ok(Object::Boolean(l == r)),
                 _ => Ok(Object::Float(l + r)), // Mock
            },
            (Object::Tensor(l), Object::Tensor(r)) => match operator {
                InfixOperator::Plus => {
                    let res = l.add(&r)?;
                    Ok(Object::Tensor(res))
                },
                InfixOperator::MatrixMultiply => {
                    let res = l.matmul(&r)?;
                    Ok(Object::Tensor(res))
                },
                _ => Err(format!("Unsupported operator for tensors: {:?}", operator)),
            },
            // Handle mixed types?
            (l, r) => Err(format!("Type mismatch: {} {:?} {}", l, operator, r)),
        }
    }

    fn apply_function(&mut self, func: Object, args: Vec<Object>) -> Result<Object, String> {
        match func {
            Object::Function { params, body, env } => {
                let extended_env = Environment::new_enclosed(env);
                let extended_env_rc = Rc::new(RefCell::new(extended_env));
                
                for (param, arg) in params.iter().zip(args) {
                    extended_env_rc.borrow_mut().set(param.clone(), arg);
                }
                
                let evaluated = self.eval_block(body, extended_env_rc)?;
                if let Object::ReturnValue(val) = evaluated {
                    Ok(*val)
                } else {
                    Ok(evaluated)
                }
            },
            Object::NativeFn(func) => func(args),
            Object::PyObject(py_obj) => {
                pyo3::Python::with_gil(|py| {
                    let args_vec: Vec<pyo3::Py<pyo3::types::PyAny>> = args.iter().map(|arg| {
                        match arg {
                            Object::Integer(i) => (*i).into_pyobject(py).unwrap().as_any().clone().unbind(),
                            Object::Float(f) => (*f).into_pyobject(py).unwrap().as_any().clone().unbind(),
                            Object::String(s) => s.into_pyobject(py).unwrap().as_any().clone().unbind(),
                            Object::Boolean(b) => (*b).into_pyobject(py).unwrap().as_any().clone().unbind(),
                            Object::PyObject(p) => p.clone_ref(py),
                            _ => py.None(),
                        }
                    }).collect();
                    
                    let py_args = pyo3::types::PyTuple::new(py, args_vec).unwrap();
                    let res = py_obj.call1(py, py_args)
                        .map_err(|e| format!("Python call error: {}", e))?;
                    
                    // Convert result back to Object
                    if let Ok(i) = res.extract::<i64>(py) {
                        Ok(Object::Integer(i))
                    } else if let Ok(f) = res.extract::<f64>(py) {
                        Ok(Object::Float(f))
                    } else if let Ok(s) = res.extract::<String>(py) {
                        Ok(Object::String(s))
                    } else if let Ok(b) = res.extract::<bool>(py) {
                        Ok(Object::Boolean(b))
                    } else {
                        Ok(Object::PyObject(res.into()))
                    }
                })
            },
            _ => Err(format!("Not a function: {}", func)),
        }
    }

    fn is_truthy(&self, obj: Object) -> bool {
        match obj {
            Object::Null => false,
            Object::Boolean(b) => b,
            Object::Integer(i) => i != 0,
            _ => true,
        }
    }
}
