use crate::ast::{Statement, StatementKind, Expression, ExpressionKind, Block, PrefixOperator, InfixOperator};
use crate::span::Span;
use crate::object::Object;
use crate::environment::Environment;
use crate::error::FluxError;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use pyo3::prelude::*;

pub struct Interpreter {
    pub modules: HashMap<String, Object>,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            modules: HashMap::new(),
        }
    }

    pub fn eval(&mut self, node: Statement, env: Rc<RefCell<Environment>>) -> Result<Object, FluxError> {
        let span = node.span;
        match node.kind {
            StatementKind::Expression(expr) => self.eval_expression(expr, env),
            StatementKind::Let { name, value, type_hint: _ } => {
                let val = self.eval_expression(value, env.clone())?;
                env.borrow_mut().set(name, val.clone());
                Ok(val)
            },
            StatementKind::Return(expr_opt) => {
                let val = match expr_opt {
                    Some(expr) => self.eval_expression(expr, env)?,
                    None => Object::Null,
                };
                Ok(Object::ReturnValue(Box::new(val)))
            },
            StatementKind::FunctionDef { name, params, body, return_type: _ } => {
                let func = Object::Function { 
                    params: params.into_iter().map(|(n, _)| n).collect(), 
                    body, 
                    env: env.clone() 
                };
                env.borrow_mut().set(name, func.clone());
                Ok(Object::Null)
            },
            StatementKind::If { condition, consequence, elif_branches, alternative } => {
                let cond = self.eval_expression(condition, env.clone())?;
                if self.is_truthy(cond) {
                    self.eval_block(consequence, env)
                } else {
                    let mut handled = false;
                    let mut result = Ok(Object::Null);
                    for (elif_cond_expr, elif_body) in elif_branches {
                        let elif_cond = self.eval_expression(elif_cond_expr, env.clone())?;
                        if self.is_truthy(elif_cond) {
                            result = self.eval_block(elif_body, env.clone());
                            handled = true;
                            break;
                        }
                    }
                    if handled {
                        result
                    } else if let Some(alt) = alternative {
                        self.eval_block(alt, env)
                    } else {
                        Ok(Object::Null)
                    }
                }
            },
            StatementKind::While { condition, body } => {
                loop {
                    let cond = self.eval_expression(condition.clone(), env.clone())?;
                    if !self.is_truthy(cond) {
                        break;
                    }
                    let result = self.eval_block(body.clone(), env.clone())?;
                    match result {
                        Object::Break => break,
                        Object::Continue => continue,
                        Object::ReturnValue(_) => return Ok(result),
                        _ => {}
                    }
                }
                Ok(Object::Null)
            },
            StatementKind::Print(expressions) => {
                let mut vals = Vec::new();
                for expr in expressions {
                    vals.push(self.eval_expression(expr, env.clone())?.to_string());
                }
                println!("{}", vals.join(" "));
                Ok(Object::Null)
            },
            StatementKind::For { variable, iterable, body } => {
                let iter_obj = self.eval_expression(iterable, env.clone())?;
                let elements = match iter_obj {
                    Object::List(l) => l.borrow().clone(),
                    Object::Tensor(_t) => {
                        return Err(FluxError::new_runtime(format!("Tensor iteration not yet implemented"), span));
                    },
                    _ => return Err(FluxError::new_runtime(format!("Cannot iterate over {}", iter_obj), span)),
                };

                for el in elements {
                    env.borrow_mut().set(variable.clone(), el);
                    let result = self.eval_block(body.clone(), env.clone())?;
                    match result {
                        Object::Break => break,
                        Object::Continue => continue,
                        Object::ReturnValue(_) => return Ok(result),
                        _ => {}
                    }
                }
                Ok(Object::Null)
            },
            StatementKind::IndexAssign { object, index, value } => {
                let obj_expr_clone = object.clone(); // Clone for potential write-back
                let obj = self.eval_expression(object, env.clone())?;
                let idx = self.eval_expression(index, env.clone())?;
                let val = self.eval_expression(value, env.clone())?;

                match obj {
                    Object::List(l) => {
                        let mut list_borrow = l.borrow_mut();
                        let mut i = match idx {
                            Object::Integer(i) => i,
                            _ => return Err(FluxError::new_runtime(format!("Index must be integer, got {}", idx), span)),
                        };
                        if i < 0 { i += list_borrow.len() as i64; }
                        if i < 0 || i >= list_borrow.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        list_borrow[i as usize] = val;
                    },
                    Object::Tensor(t) => {
                          let mut t_inner = t.clone();
                          let mut i = match idx {
                              Object::Integer(i) => i,
                              _ => return Err(FluxError::new_runtime(format!("Index must be integer, got {}", idx), span)),
                          };
                          let val_f = match val {
                              Object::Float(f) => f,
                              Object::Integer(i) => i as f64,
                              _ => return Err(FluxError::new_runtime(format!("Tensor value must be numeric, got {}", val), span)),
                          };
                          
                          if t_inner.inner.ndim() != 1 {
                              return Err(FluxError::new_runtime(format!("Tensor index assignment currently only supported for 1D tensors"), span));
                          }
                          if i < 0 { i += t_inner.inner.len() as i64; }
                          if i < 0 || i >= t_inner.inner.len() as i64 {
                              return Err(FluxError::new_runtime(format!("Index out of bounds for tensor: {}", i), span));
                          }
                          t_inner.inner[i as usize] = val_f;

                          // Write back for Tensor
                         if let ExpressionKind::Identifier(name) = obj_expr_clone.kind {
                            env.borrow_mut().set(name, Object::Tensor(t_inner));
                         } else {
                            return Err(FluxError::new_runtime(format!("Nested tensor index assignment not yet supported"), span));
                         }
                    }
                    Object::Dictionary(d) => {
                        d.borrow_mut().insert(idx, val);
                    }
                    _ => return Err(FluxError::new_runtime(format!("Cannot assign to index of {}", obj), span)),
                }
                Ok(Object::Null)
            },
            StatementKind::Break => Ok(Object::Break),
            StatementKind::Continue => Ok(Object::Continue),
            StatementKind::Import { path, alias } => {
                let module_name = if let Some(a) = &alias {
                    a.clone()
                } else {
                    std::path::Path::new(&path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or(path.clone())
                };

                if let Some(module) = self.modules.get(&path) {
                    env.borrow_mut().set(module_name, module.clone());
                    return Ok(Object::Null);
                }

                let file_path = if path.ends_with(".fl") {
                    path.clone()
                } else {
                    format!("{}.fl", path)
                };

                let content_res = std::fs::read_to_string(&file_path);
                
                if let Ok(content) = content_res {
                    let lexer = crate::lexer::Lexer::new(&content);
                    let mut parser = crate::parser::Parser::new(lexer);
                    let program = parser.parse_program();

                    if !parser.errors.is_empty() {
                        return Err(FluxError::new_parse(parser.errors[0].message.clone(), parser.errors[0].span));
                    }

                    let module_env = Rc::new(RefCell::new(Environment::new()));
                    for stmt in program {
                        self.eval(stmt, module_env.clone())?;
                    }

                    let module = Object::Module { name: module_name.clone(), env: module_env };
                    self.modules.insert(path.clone(), module.clone());
                    env.borrow_mut().set(module_name, module);
                } else {
                    // Try as Python module
                    let py_module = pyo3::Python::attach(|py| {
                        pyo3::types::PyModule::import(py, path.as_str())
                            .map(|m| Object::PyObject(m.into()))
                            .map_err(|e| FluxError::new_runtime(format!("Failed to import Flux or Python module '{}': {}", path, e), span))
                    })?;
                    self.modules.insert(path.clone(), py_module.clone());
                    env.borrow_mut().set(module_name, py_module);
                }

                Ok(Object::Null)
            },
            StatementKind::Assert { condition, message } => {
                let cond = self.eval_expression(condition, env.clone())?;
                if !self.is_truthy(cond) {
                    let msg = if let Some(m_expr) = message {
                        self.eval_expression(m_expr, env.clone())?.to_string()
                    } else {
                        "Assertion failed".to_string()
                    };
                    return Err(FluxError::new_runtime(msg, span));
                }
                Ok(Object::Null)
            }
        }
    }

    pub fn eval_block(&mut self, block: Block, env: Rc<RefCell<Environment>>) -> Result<Object, FluxError> {
        let mut result = Object::Null;
        for stmt in block.statements {
            result = self.eval(stmt, env.clone())?;
            match result {
                Object::ReturnValue(_) | Object::Break | Object::Continue => return Ok(result),
                _ => {}
            }
        }
        Ok(result)
    }

    fn eval_expression(&mut self, expr: Expression, env: Rc<RefCell<Environment>>) -> Result<Object, FluxError> {
        let span = expr.span;
        match expr.kind {
            ExpressionKind::Integer(i) => Ok(Object::Integer(i)),
            ExpressionKind::Float(f) => Ok(Object::Float(f)),
            ExpressionKind::String(s) => Ok(Object::String(s)),
            ExpressionKind::Identifier(name) => {
                match env.borrow().get(&name) {
                    Some(obj) => Ok(obj),
                    None => Err(FluxError::new_runtime(format!("Identifier not found: {}", name), span)),
                }
            },
            ExpressionKind::Prefix { operator, right } => {
                let right_val = self.eval_expression(*right, env)?;
                self.eval_prefix_expression(operator, right_val).map_err(|e| FluxError::new_runtime(e, span))
            },
            ExpressionKind::Infix { left, operator, right } => {
                let left_val = self.eval_expression(*left, env.clone())?;
                let right_val = self.eval_expression(*right, env)?;
                self.eval_infix_expression(operator, left_val, right_val).map_err(|e| FluxError::new_runtime(e, span))
            },
            ExpressionKind::Call { function, arguments } => {
                let func = self.eval_expression(*function, env.clone())?;
                let args = arguments.into_iter()
                    .map(|arg| self.eval_expression(arg, env.clone()))
                    .collect::<Result<Vec<Object>, FluxError>>()?;
                self.apply_function(func, args, span)
            },
            ExpressionKind::MethodCall { object, method, arguments } => {
                let obj = self.eval_expression(*object, env.clone())?;
                
                // If it's a PyObject, we have to decide between Flux-style call (method(obj, args))
                // and Python-style call (obj.method(args)).
                if let Object::PyObject(ref py_obj) = obj {
                    // Try Flux-style first if method exists in env
                    if let Some(func) = env.borrow().get(&method) {
                        let mut args = vec![obj.clone()];
                        for arg_expr in arguments.clone() {
                            args.push(self.eval_expression(arg_expr, env.clone())?);
                        }
                        let res = self.apply_function(func, args, span);
                        if res.is_ok() {
                            return res;
                        }
                        // If it failed (e.g. signature mismatch), fall back to Python-style logic below
                    }
                    
                    // Python-style call (obj.method(args))
                    return pyo3::Python::attach(|py| {
                        let attr = py_obj.bind(py).getattr(method.as_str())
                            .map_err(|e| FluxError::new_runtime(format!("Method '{}' found in Flux but failed call AND Python getattr also failed: {}", method, e), span))?;
                        
                        let mut py_args = Vec::new();
                        for arg_expr in arguments {
                            let arg_val = self.eval_expression(arg_expr, env.clone())?;
                            py_args.push(self.flux_to_py(py, arg_val, span)?);
                        }
                        
                        let py_args_tuple = pyo3::types::PyTuple::new(py, py_args)
                            .map_err(|e| FluxError::new_runtime(format!("Failed to create Python tuple for method call: {}", e), span))?;
                        
                        let res = attr.call1(py_args_tuple)
                            .map_err(|e| FluxError::new_runtime(format!("Python method call error: {}", e), span))?;
                        
                        self.py_to_flux(py, res, span)
                    });
                }

                // Normal Flux object
                match env.borrow().get(&method) {
                    Some(func) => {
                        let mut args = vec![obj];
                        for arg in arguments {
                            args.push(self.eval_expression(arg, env.clone())?);
                        }
                        self.apply_function(func, args, span)
                    }
                    None => Err(FluxError::new_runtime(format!("Method '{}' not found", method), span))
                }
            },
            ExpressionKind::Get { object, name } => {
                let obj = self.eval_expression(*object, env)?;
                match obj {
                    Object::PyObject(py_obj) => {
                        pyo3::Python::attach(|py| {
                            let getattr = py_obj.bind(py).getattr(name.as_str())
                                .map_err(|e| FluxError::new_runtime(format!("Python getattr error: {}", e), span))?;
                            self.py_to_flux(py, getattr, span)
                        })
                    },
                    Object::Module { name: _, env: module_env } => {
                        match module_env.borrow().get(&name) {
                            Some(v) => Ok(v),
                            None => Err(FluxError::new_runtime(format!("Member '{}' not found in module", name), span)),
                        }
                    },
                    _ => Err(FluxError::new_runtime(format!("Property access not supported on {}", obj), span)),
                }
            },
            ExpressionKind::List(elements) => {
                let mut objs = Vec::new();
                for el in elements {
                    objs.push(self.eval_expression(el, env.clone())?);
                }
                Ok(Object::List(Rc::new(RefCell::new(objs))))
            },
            ExpressionKind::Index { object, index } => {
                let obj = self.eval_expression(*object, env.clone())?;
                let idx = self.eval_expression(*index, env)?;
                
                match (obj, idx) {
                    (Object::List(l), Object::Integer(mut i)) => {
                        let list_borrow = l.borrow();
                        if i < 0 { i += list_borrow.len() as i64; }
                        if i < 0 || i >= list_borrow.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        Ok(list_borrow[i as usize].clone())
                    },
                    (Object::List(l), Object::Slice { start, stop, step }) => {
                        let list_borrow = l.borrow();
                        let len = list_borrow.len() as i64;
                        let abs_start = match start {
                            Some(v) => if v < 0 { v + len } else { v },
                            None => if step > 0 { 0 } else { len - 1 }
                        };
                        let abs_stop = match stop {
                            Some(v) => if v < 0 { v + len } else { v },
                            None => if step > 0 { len } else { -1 } 
                        };

                        let mut result_list = Vec::new();
                        let mut curr = abs_start;
                        
                        if step > 0 {
                            if curr < 0 { curr = 0; }
                            if curr > len { curr = len; }
                            let mut stop_bound = abs_stop;
                            if stop_bound < 0 { stop_bound = 0; }
                            if stop_bound > len { stop_bound = len; }
                            
                            while curr < stop_bound {
                                if curr < len {
                                    result_list.push(list_borrow[curr as usize].clone());
                                }
                                curr += step;
                            }
                        } else {
                            if curr >= len { curr = len - 1; }
                            // curr logic for negative step:
                            // we loop while curr > abs_stop.
                            // abs_stop can be -1.
                            
                            let stop_bound = abs_stop;
                            
                            while curr > stop_bound {
                                if curr >= 0 && curr < len {
                                    result_list.push(list_borrow[curr as usize].clone());
                                }
                                curr += step;
                            }
                        }
                        
                        Ok(Object::List(Rc::new(RefCell::new(result_list))))
                    },
                    (Object::String(s), Object::Integer(mut i)) => {
                        if i < 0 { i += s.len() as i64; }
                        if i < 0 || i >= s.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        let ch = s.chars().nth(i as usize).unwrap();
                        Ok(Object::String(ch.to_string()))
                    },
                    (Object::String(s), Object::Slice { start, stop, step }) => {
                        let chars: Vec<char> = s.chars().collect();
                        let len = chars.len() as i64;
                        
                        let abs_start = match start {
                            Some(v) => if v < 0 { v + len } else { v },
                            None => if step > 0 { 0 } else { len - 1 }
                        };
                        let abs_stop = match stop {
                            Some(v) => if v < 0 { v + len } else { v },
                            None => if step > 0 { len } else { -1 } 
                        };

                        let mut result_chars = Vec::new();
                        let mut curr = abs_start;
                        
                        if step > 0 {
                            if curr < 0 { curr = 0; }
                            if curr > len { curr = len; }
                            let mut stop_bound = abs_stop;
                            if stop_bound < 0 { stop_bound = 0; }
                            if stop_bound > len { stop_bound = len; }
                            
                            while curr < stop_bound {
                                if curr < len {
                                    result_chars.push(chars[curr as usize]);
                                }
                                curr += step;
                            }
                        } else {
                            if curr >= len { curr = len - 1; }
                            
                            let stop_bound = abs_stop;
                            
                            while curr > stop_bound {
                                if curr >= 0 && curr < len {
                                    result_chars.push(chars[curr as usize]);
                                }
                                curr += step;
                            }
                        }
                        Ok(Object::String(result_chars.into_iter().collect()))
                    },
                    (Object::Tensor(t), Object::Integer(mut i)) => {
                        if t.inner.ndim() == 0 {
                            return Err(FluxError::new_runtime(format!("Cannot index 0D tensor"), span));
                        }
                        let shape = t.inner.shape();
                        if i < 0 { i += shape[0] as i64; }
                        if i < 0 || i >= shape[0] as i64 {
                             return Err(FluxError::new_runtime(format!("Index out of bounds: {} (shape: {:?})", i, shape), span));
                        }
                        
                        if t.inner.ndim() == 1 {
                            Ok(Object::Float(t.inner[i as usize]))
                        } else {
                            let sub = t.inner.index_axis(ndarray::Axis(0), i as usize);
                            Ok(Object::Tensor(crate::tensor::Tensor { inner: sub.to_owned() }))
                        }
                    },
                    (Object::Dictionary(d), k) => {
                        let dict = d.borrow();
                        match dict.get(&k) {
                            Some(v) => Ok(v.clone()),
                            None => Err(FluxError::new_runtime(format!("Key not found: {}", k), span)),
                        }
                    },
                    (o, i) => Err(FluxError::new_runtime(format!("Cannot index {} with {}", o, i), span)),
                }
            },
            ExpressionKind::ListComprehension { element, variable, iterable, condition } => {
                let iter_obj = self.eval_expression(*iterable, env.clone())?;
                let elements = match iter_obj {
                    Object::List(l) => l.borrow().clone(),
                    _ => return Err(FluxError::new_runtime(format!("Cannot iterate over {} in list comprehension", iter_obj), span)),
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
                 Ok(Object::List(Rc::new(RefCell::new(result_list))))
            },
            ExpressionKind::Dictionary(pairs) => {
                let mut map = std::collections::HashMap::new();
                for (key_expr, val_expr) in pairs {
                    let key = self.eval_expression(key_expr, env.clone())?;
                    let val = self.eval_expression(val_expr, env.clone())?;
                    map.insert(key, val);
                }
                Ok(Object::Dictionary(Rc::new(RefCell::new(map))))
            },
            ExpressionKind::Slice { start, end, step } => {
                let s = if let Some(expr) = start {
                     match self.eval_expression(*expr, env.clone())? {
                         Object::Integer(i) => Some(i),
                         _ => return Err(FluxError::new_runtime("Slice start must be integer".to_string(), span)),
                     }
                } else { None };
                
                let e = if let Some(expr) = end {
                     match self.eval_expression(*expr, env.clone())? {
                         Object::Integer(i) => Some(i),
                         _ => return Err(FluxError::new_runtime("Slice end must be integer".to_string(), span)),
                     }
                } else { None };
                
                let st = if let Some(expr) = step {
                     match self.eval_expression(*expr, env.clone())? {
                         Object::Integer(i) => i,
                         _ => return Err(FluxError::new_runtime("Slice step must be integer".to_string(), span)),
                     }
                } else { 1 };
                
                if st == 0 {
                    return Err(FluxError::new_runtime("Slice step cannot be zero".to_string(), span));
                }

                Ok(Object::Slice { start: s, stop: e, step: st })
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
            PrefixOperator::BitwiseNot => match right {
                Object::Integer(i) => Ok(Object::Integer(!i)),
                _ => Err(format!("Bitwise NOT not supported for {}", right)),
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
                InfixOperator::Power => {
                    if r >= 0 {
                        Ok(Object::Integer(l.pow(r as u32)))
                    } else {
                        Ok(Object::Float((l as f64).powf(r as f64)))
                    }
                },
                InfixOperator::BitwiseAnd => Ok(Object::Integer(l & r)),
                InfixOperator::BitwiseOr => Ok(Object::Integer(l | r)),
                InfixOperator::BitwiseXor => Ok(Object::Integer(l ^ r)),
                InfixOperator::ShiftLeft => {
                    if r < 0 { return Err(format!("Negative shift count: {}", r)); }
                    if r >= 64 { return Ok(Object::Integer(0)); }
                    Ok(Object::Integer(l << r))
                },
                InfixOperator::ShiftRight => {
                    if r < 0 { return Err(format!("Negative shift count: {}", r)); }
                    if r >= 64 { return Ok(Object::Integer(if l >= 0 { 0 } else { -1 })); }
                    Ok(Object::Integer(l >> r))
                },
                _ => Err(format!("Unknown operator for integers")),
            },
            (Object::Float(l), Object::Float(r)) => match operator {
                  InfixOperator::Plus => Ok(Object::Float(l + r)),
                  InfixOperator::Minus => Ok(Object::Float(l - r)),
                  InfixOperator::Multiply => Ok(Object::Float(l * r)),
                  InfixOperator::Divide => Ok(Object::Float(l / r)),
                  InfixOperator::Equal => Ok(Object::Boolean(l == r)),
                  InfixOperator::NotEqual => Ok(Object::Boolean(l != r)),
                  InfixOperator::LessThan => Ok(Object::Boolean(l < r)),
                  InfixOperator::GreaterThan => Ok(Object::Boolean(l > r)),
                  InfixOperator::LessThanOrEqual => Ok(Object::Boolean(l <= r)),
                  InfixOperator::GreaterThanOrEqual => Ok(Object::Boolean(l >= r)),
                  InfixOperator::Power => Ok(Object::Float(l.powf(r))),
                  _ => Err(format!("Unsupported operator for floats: {:?}", operator)),
            },
            (Object::Float(l), Object::Integer(r)) => match operator {
                InfixOperator::Plus => Ok(Object::Float(l + r as f64)),
                InfixOperator::Minus => Ok(Object::Float(l - r as f64)),
                InfixOperator::Multiply => Ok(Object::Float(l * r as f64)),
                InfixOperator::Divide => Ok(Object::Float(l / r as f64)),
                InfixOperator::Equal => Ok(Object::Boolean(l == r as f64)),
                InfixOperator::NotEqual => Ok(Object::Boolean(l != r as f64)),
                InfixOperator::LessThan => Ok(Object::Boolean(l < r as f64)),
                InfixOperator::GreaterThan => Ok(Object::Boolean(l > r as f64)),
                InfixOperator::LessThanOrEqual => Ok(Object::Boolean(l <= r as f64)),
                InfixOperator::GreaterThanOrEqual => Ok(Object::Boolean(l >= r as f64)),
                InfixOperator::Power => Ok(Object::Float(l.powf(r as f64))),
                _ => Err(format!("Unsupported operator for float and integer: {:?}", operator)),
            },
            (Object::Integer(l), Object::Float(r)) => match operator {
                InfixOperator::Plus => Ok(Object::Float(l as f64 + r)),
                InfixOperator::Minus => Ok(Object::Float(l as f64 - r)),
                InfixOperator::Multiply => Ok(Object::Float(l as f64 * r)),
                InfixOperator::Divide => Ok(Object::Float(l as f64 / r)),
                InfixOperator::Equal => Ok(Object::Boolean(l as f64 == r)),
                InfixOperator::NotEqual => Ok(Object::Boolean(l as f64 != r)),
                InfixOperator::LessThan => Ok(Object::Boolean((l as f64) < r)),
                InfixOperator::GreaterThan => Ok(Object::Boolean((l as f64) > r)),
                InfixOperator::LessThanOrEqual => Ok(Object::Boolean((l as f64) <= r)),
                InfixOperator::GreaterThanOrEqual => Ok(Object::Boolean((l as f64) >= r)),
                InfixOperator::Power => Ok(Object::Float((l as f64).powf(r))),
                _ => Err(format!("Unsupported operator for integer and float: {:?}", operator)),
            },
            (Object::Tensor(l), Object::Tensor(r)) => match operator {
                InfixOperator::Plus => {
                    let res = l.add(&r)?;
                    Ok(Object::Tensor(res))
                },
                InfixOperator::Minus => {
                    let res = l.sub(&r)?;
                    Ok(Object::Tensor(res))
                },
                InfixOperator::Multiply => {
                    let res = l.mul(&r)?;
                    Ok(Object::Tensor(res))
                },
                InfixOperator::Divide => {
                    let res = l.div(&r)?;
                    Ok(Object::Tensor(res))
                },
                InfixOperator::MatrixMultiply => {
                    let res = l.matmul(&r)?;
                    Ok(Object::Tensor(res))
                },
                _ => Err(format!("Unsupported operator for tensors: {:?}", operator)),
            },
            (Object::Tensor(t), Object::Float(f)) => match operator {
                InfixOperator::Plus => Ok(Object::Tensor(t.add_scalar(f))),
                InfixOperator::Minus => Ok(Object::Tensor(t.sub_scalar(f))),
                InfixOperator::Multiply => Ok(Object::Tensor(t.mul_scalar(f))),
                InfixOperator::Divide => Ok(Object::Tensor(t.div_scalar(f))),
                _ => Err(format!("Unsupported operator for tensor and float: {:?}", operator)),
            },
            (Object::Tensor(t), Object::Integer(i)) => match operator {
                InfixOperator::Plus => Ok(Object::Tensor(t.add_scalar(i as f64))),
                InfixOperator::Minus => Ok(Object::Tensor(t.sub_scalar(i as f64))),
                InfixOperator::Multiply => Ok(Object::Tensor(t.mul_scalar(i as f64))),
                InfixOperator::Divide => Ok(Object::Tensor(t.div_scalar(i as f64))),
                _ => Err(format!("Unsupported operator for tensor and integer: {:?}", operator)),
            },
            (Object::Float(f), Object::Tensor(t)) => match operator {
                InfixOperator::Plus => Ok(Object::Tensor(t.add_scalar(f))),
                InfixOperator::Multiply => Ok(Object::Tensor(t.mul_scalar(f))),
                _ => Err(format!("Unsupported operator for float and tensor: {:?}", operator)),
            },
            (Object::Integer(i), Object::Tensor(t)) => match operator {
                InfixOperator::Plus => Ok(Object::Tensor(t.add_scalar(i as f64))),
                InfixOperator::Multiply => Ok(Object::Tensor(t.mul_scalar(i as f64))),
                _ => Err(format!("Unsupported operator for integer and tensor: {:?}", operator)),
            },
            (Object::String(l), Object::String(r)) => match operator {
                InfixOperator::Plus => Ok(Object::String(format!("{}{}", l, r))),
                InfixOperator::Equal => Ok(Object::Boolean(l == r)),
                InfixOperator::NotEqual => Ok(Object::Boolean(l != r)),
                InfixOperator::In => Ok(Object::Boolean(r.contains(&l))),
                InfixOperator::NotIn => Ok(Object::Boolean(!r.contains(&l))),
                _ => Err(format!("Unsupported operator for strings: {:?}", operator)),
            },
            (Object::String(s), Object::Integer(i)) => match operator {
                InfixOperator::Multiply => {
                    if i < 0 { return Err(format!("Negative string multiplication count: {}", i)); }
                    Ok(Object::String(s.repeat(i as usize)))
                },
                _ => Err(format!("Unsupported operator for string and integer: {:?}", operator)),
            },
            (Object::Integer(i), Object::String(s)) => match operator {
                InfixOperator::Multiply => {
                    if i < 0 { return Err(format!("Negative string multiplication count: {}", i)); }
                    Ok(Object::String(s.repeat(i as usize)))
                },
                _ => Err(format!("Unsupported operator for integer and string: {:?}", operator)),
            },
            (Object::List(l1), Object::List(l2)) => match operator {
                InfixOperator::Plus => {
                    let mut res = l1.borrow().clone();
                    res.extend(l2.borrow().clone());
                    Ok(Object::List(Rc::new(RefCell::new(res))))
                },
                InfixOperator::Equal => Ok(Object::Boolean(*l1.borrow() == *l2.borrow())),
                InfixOperator::NotEqual => Ok(Object::Boolean(*l1.borrow() != *l2.borrow())),
                _ => Err(format!("Unsupported operator for lists: {:?}", operator)),
            },
            (Object::List(l), Object::Integer(i)) => match operator {
                InfixOperator::Multiply => {
                    if i < 0 { return Err(format!("Negative list multiplication count: {}", i)); }
                    let mut res = Vec::new();
                    let list_borrow = l.borrow();
                    for _ in 0..i {
                        res.extend(list_borrow.clone());
                    }
                    Ok(Object::List(Rc::new(RefCell::new(res))))
                },
                _ => Err(format!("Unsupported operator for list and integer: {:?}", operator)),
            },
            (l, Object::List(r)) => match operator {
                InfixOperator::In => Ok(Object::Boolean(r.borrow().contains(&l))),
                InfixOperator::NotIn => Ok(Object::Boolean(!r.borrow().contains(&l))),
                InfixOperator::Multiply => {
                    if let Object::Integer(i) = l {
                        if i < 0 { return Err(format!("Negative list multiplication count: {}", i)); }
                        let mut res = Vec::new();
                        let list_borrow = r.borrow();
                        for _ in 0..i {
                            res.extend(list_borrow.clone());
                        }
                        return Ok(Object::List(Rc::new(RefCell::new(res))));
                    }
                    Err(format!("Unsupported operator for {} and list: {:?}", l, operator))
                }
                _ => Err(format!("Unsupported operator for list: {:?}", operator)),
            },
            (Object::Boolean(l), Object::Boolean(r)) => match operator {
                InfixOperator::Equal => Ok(Object::Boolean(l == r)),
                InfixOperator::NotEqual => Ok(Object::Boolean(l != r)),
                _ => Err(format!("Unsupported operator for booleans: {:?}", operator)),
            },
            (Object::Null, Object::Null) => match operator {
                InfixOperator::Equal => Ok(Object::Boolean(true)),
                InfixOperator::NotEqual => Ok(Object::Boolean(false)),
                _ => Err(format!("Unsupported operator for null: {:?}", operator)),
            },
            // Handle mixed types?
            (l, r) => Err(format!("Type mismatch: {} {:?} {}", l, operator, r)),
        }
    }

    fn apply_function(&mut self, func: Object, args: Vec<Object>, span: Span) -> Result<Object, FluxError> {
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
            Object::NativeFn(func) => func(args).map_err(|e| FluxError::new_runtime(e, span)),
            Object::PyObject(py_obj) => {
                pyo3::Python::attach(|py| {
                    let mut py_args = Vec::new();
                    for arg in args {
                        py_args.push(self.flux_to_py(py, arg, span)?);
                    }
                    
                    let py_args_tuple = pyo3::types::PyTuple::new(py, py_args)
                        .map_err(|e| FluxError::new_runtime(format!("Failed to create Python tuple: {}", e), span))?;
                    
                    let res = py_obj.bind(py).call1(py_args_tuple)
                        .map_err(|e| FluxError::new_runtime(format!("Python call error: {}", e), span))?;
                    
                    self.py_to_flux(py, res, span)
                })
            },
            _ => Err(FluxError::new_runtime(format!("Not a function: {}", func), span)),
        }
    }

    fn flux_to_py<'py>(&self, py: Python<'py>, obj: Object, span: Span) -> Result<Bound<'py, pyo3::PyAny>, FluxError> {
        match obj {
            Object::Integer(i) => Ok(i.into_pyobject(py).map_err(|e| FluxError::new_runtime(e.to_string(), span))?.as_any().clone()),
            Object::Float(f) => Ok(f.into_pyobject(py).map_err(|e| FluxError::new_runtime(e.to_string(), span))?.as_any().clone()),
            Object::String(s) => Ok(s.into_pyobject(py).map_err(|e| FluxError::new_runtime(e.to_string(), span))?.as_any().clone()),
            Object::Boolean(b) => Ok(pyo3::types::PyBool::new(py, b).as_any().clone()),
            Object::Null => Ok(py.None().into_bound(py)),
            Object::List(l) => {
                let py_list = pyo3::types::PyList::empty(py);
                for item in l.borrow().iter() {
                    py_list.append(self.flux_to_py(py, item.clone(), span)?)
                        .map_err(|e| FluxError::new_runtime(format!("Failed to append to Python list: {}", e), span))?;
                }
                Ok(py_list.into_any())
            },
            Object::Dictionary(d) => {
                let py_dict = pyo3::types::PyDict::new(py);
                for (k, v) in d.borrow().iter() {
                    py_dict.set_item(self.flux_to_py(py, k.clone(), span)?, self.flux_to_py(py, v.clone(), span)?)
                        .map_err(|e| FluxError::new_runtime(format!("Failed to set item in Python dict: {}", e), span))?;
                }
                Ok(py_dict.into_any())
            },
            Object::Tensor(t) => {
                // If numpy is available, convert to numpy array
                let np = pyo3::types::PyModule::import(py, "numpy")
                    .map_err(|e| FluxError::new_runtime(format!("Failed to import numpy for tensor conversion: {}", e), span))?;
                
                let data = t.inner.as_slice().unwrap().to_vec();
                let shape = t.inner.shape().to_vec();
                
                let py_data = data.into_pyobject(py).unwrap();
                let py_shape = shape.into_pyobject(py).unwrap();
                
                let array = np.call_method1("array", (py_data,))
                    .map_err(|e| FluxError::new_runtime(format!("Numpy array creation failed: {}", e), span))?;
                let reshaped = array.call_method1("reshape", (py_shape,))
                    .map_err(|e| FluxError::new_runtime(format!("Numpy reshape failed: {}", e), span))?;
                
                Ok(reshaped)
            },
            Object::PyObject(p) => Ok(p.into_bound(py)),
            _ => Err(FluxError::new_runtime(format!("Cannot convert {} to Python", obj), span)),
        }
    }

    fn py_to_flux<'py>(&self, py: Python<'py>, obj: Bound<'py, pyo3::PyAny>, span: Span) -> Result<Object, FluxError> {
        if let Ok(i) = obj.extract::<i64>() {
            Ok(Object::Integer(i))
        } else if let Ok(f) = obj.extract::<f64>() {
            Ok(Object::Float(f))
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(Object::String(s))
        } else if let Ok(b) = obj.extract::<bool>() {
            Ok(Object::Boolean(b))
        } else if obj.is_none() {
            Ok(Object::Null)
        } else if obj.is_instance_of::<pyo3::types::PyList>() {
            let py_list = obj.cast::<pyo3::types::PyList>().unwrap();
            let mut flux_list = Vec::new();
            for item in py_list.iter() {
                flux_list.push(self.py_to_flux(py, item, span)?);
            }
            Ok(Object::List(Rc::new(RefCell::new(flux_list))))
        } else if obj.is_instance_of::<pyo3::types::PyDict>() {
            let py_dict = obj.cast::<pyo3::types::PyDict>().unwrap();
            let mut flux_dict = std::collections::HashMap::new();
            for (k, v) in py_dict.iter() {
                flux_dict.insert(self.py_to_flux(py, k, span)?, self.py_to_flux(py, v, span)?);
            }
            Ok(Object::Dictionary(Rc::new(RefCell::new(flux_dict))))
        } else {
            // Check if it's a numpy array
            let np_res = pyo3::types::PyModule::import(py, "numpy");
            if let Ok(np) = np_res {
                let is_ndarray = obj.is_instance(np.getattr("ndarray").unwrap().as_ref()).unwrap_or(false);
                if is_ndarray {
                    let flatten = obj.call_method0("flatten")
                        .map_err(|e| FluxError::new_runtime(format!("Numpy flatten failed: {}", e), span))?;
                    let data: Vec<f64> = flatten.extract()
                        .map_err(|e| FluxError::new_runtime(format!("Failed to extract numpy data: {}", e), span))?;
                    let shape_obj = obj.getattr("shape")
                        .map_err(|e| FluxError::new_runtime(format!("Failed to get numpy shape: {}", e), span))?;
                    let shape: Vec<usize> = shape_obj.extract()
                        .map_err(|e| FluxError::new_runtime(format!("Failed to extract numpy shape: {}", e), span))?;
                    
                    let t = crate::tensor::Tensor::new(data, shape)
                        .map_err(|e| FluxError::new_runtime(e, span))?;
                    return Ok(Object::Tensor(t));
                }
            }
            Ok(Object::PyObject(obj.into()))
        }
    }

    fn is_truthy(&self, obj: Object) -> bool {
        obj.is_truthy()
    }
}
