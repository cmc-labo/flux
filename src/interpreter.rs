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
            StatementKind::Let { name, value } => {
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
            StatementKind::FunctionDef { name, params, body } => {
                let func = Object::Function { params, body, env: env.clone() };
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
                    Object::List(l) => l,
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
                let obj_expr = object.clone();
                let obj = self.eval_expression(object, env.clone())?;
                let idx = self.eval_expression(index, env.clone())?;
                let val = self.eval_expression(value, env.clone())?;

                match obj {
                    Object::List(mut l) => {
                        let mut i = match idx {
                            Object::Integer(i) => i,
                            _ => return Err(FluxError::new_runtime(format!("Index must be integer, got {}", idx), span)),
                        };
                        if i < 0 { i += l.len() as i64; }
                        if i < 0 || i >= l.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        l[i as usize] = val;
                        
                        if let ExpressionKind::Identifier(name) = obj_expr.kind {
                            env.borrow_mut().set(name, Object::List(l));
                        } else {
                            return Err(FluxError::new_runtime(format!("Nested index assignment not yet supported"), span));
                        }
                    },
                    Object::Tensor(mut t) => {
                          let mut i = match idx {
                              Object::Integer(i) => i,
                              _ => return Err(FluxError::new_runtime(format!("Index must be integer, got {}", idx), span)),
                          };
                          let val_f = match val {
                              Object::Float(f) => f,
                              Object::Integer(i) => i as f64,
                              _ => return Err(FluxError::new_runtime(format!("Tensor value must be numeric, got {}", val), span)),
                          };
                          
                          if t.inner.ndim() != 1 {
                              return Err(FluxError::new_runtime(format!("Tensor index assignment currently only supported for 1D tensors"), span));
                          }
                          if i < 0 { i += t.inner.len() as i64; }
                          if i < 0 || i >= t.inner.len() as i64 {
                              return Err(FluxError::new_runtime(format!("Index out of bounds for tensor: {}", i), span));
                          }
                          t.inner[i as usize] = val_f;

                         if let ExpressionKind::Identifier(name) = obj_expr.kind {
                            env.borrow_mut().set(name, Object::Tensor(t));
                         } else {
                            return Err(FluxError::new_runtime(format!("Nested tensor index assignment not yet supported"), span));
                         }
                    }
                    Object::Dictionary(mut d) => {
                        d.insert(idx, val);
                        if let ExpressionKind::Identifier(name) = obj_expr.kind {
                            env.borrow_mut().set(name, Object::Dictionary(d));
                        } else {
                            return Err(FluxError::new_runtime(format!("Nested dictionary index assignment not yet supported"), span));
                        }
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

                let content = std::fs::read_to_string(&file_path)
                    .map_err(|e| FluxError::new_runtime(format!("Could not read module file {}: {}", file_path, e), span))?;

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
                self.apply_function(func, args)
            },
            ExpressionKind::Get { object, name } => {
                let obj = self.eval_expression(*object, env)?;
                match obj {
                    Object::PyObject(py_obj) => {
                        pyo3::Python::with_gil(|py| {
                            let getattr = py_obj.getattr(py, name.as_str())
                                .map_err(|e| FluxError::new_runtime(format!("Python getattr error: {}", e), span))?;
                            Ok(Object::PyObject(getattr.into()))
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
                Ok(Object::List(objs))
            },
            ExpressionKind::Index { object, index } => {
                let obj = self.eval_expression(*object, env.clone())?;
                let idx = self.eval_expression(*index, env)?;
                
                match (obj, idx) {
                    (Object::List(l), Object::Integer(mut i)) => {
                        if i < 0 { i += l.len() as i64; }
                        if i < 0 || i >= l.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        Ok(l[i as usize].clone())
                    },
                    (Object::String(s), Object::Integer(mut i)) => {
                        if i < 0 { i += s.len() as i64; }
                        if i < 0 || i >= s.len() as i64 {
                            return Err(FluxError::new_runtime(format!("Index out of bounds: {}", i), span));
                        }
                        let ch = s.chars().nth(i as usize).unwrap();
                        Ok(Object::String(ch.to_string()))
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
                        match d.get(&k) {
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
                    Object::List(l) => l,
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
                 Ok(Object::List(result_list))
            },
            ExpressionKind::Dictionary(pairs) => {
                let mut map = std::collections::HashMap::new();
                for (key_expr, val_expr) in pairs {
                    let key = self.eval_expression(key_expr, env.clone())?;
                    let val = self.eval_expression(val_expr, env.clone())?;
                    map.insert(key, val);
                }
                Ok(Object::Dictionary(map))
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
                InfixOperator::MatrixMultiply => {
                    let res = l.matmul(&r)?;
                    Ok(Object::Tensor(res))
                },
                _ => Err(format!("Unsupported operator for tensors: {:?}", operator)),
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
                    let mut res = l1.clone();
                    res.extend(l2);
                    Ok(Object::List(res))
                },
                _ => Err(format!("Unsupported operator for lists: {:?}", operator)),
            },
            (Object::List(l), Object::Integer(i)) => match operator {
                InfixOperator::Multiply => {
                    if i < 0 { return Err(format!("Negative list multiplication count: {}", i)); }
                    let mut res = Vec::new();
                    for _ in 0..i {
                        res.extend(l.clone());
                    }
                    Ok(Object::List(res))
                },
                _ => Err(format!("Unsupported operator for list and integer: {:?}", operator)),
            },
            (l, Object::List(r)) => match operator {
                InfixOperator::In => Ok(Object::Boolean(r.contains(&l))),
                InfixOperator::NotIn => Ok(Object::Boolean(!r.contains(&l))),
                InfixOperator::Multiply => {
                    if let Object::Integer(i) = l {
                        if i < 0 { return Err(format!("Negative list multiplication count: {}", i)); }
                        let mut res = Vec::new();
                        for _ in 0..i {
                            res.extend(r.clone());
                        }
                        return Ok(Object::List(res));
                    }
                    Err(format!("Unsupported operator for {} and list: {:?}", l, operator))
                }
                _ => Err(format!("Unsupported operator for list: {:?}", operator)),
            },
            // Handle mixed types?
            (l, r) => Err(format!("Type mismatch: {} {:?} {}", l, operator, r)),
        }
    }

    fn apply_function(&mut self, func: Object, args: Vec<Object>) -> Result<Object, FluxError> {
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
            Object::NativeFn(func) => func(args).map_err(|e| FluxError::new_runtime(e, Span::new(0, 0))), // Spans for builtins?
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
                        .map_err(|e| FluxError::new_runtime(format!("Python call error: {}", e), Span::new(0, 0)))?;
                    
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
            _ => Err(FluxError::new_runtime(format!("Not a function: {}", func), Span::new(0, 0))),
        }
    }

    fn is_truthy(&self, obj: Object) -> bool {
        obj.is_truthy()
    }
}
