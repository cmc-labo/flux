
mod token;
mod lexer;
mod ast;
mod span;
mod parser;
mod object;
mod environment;
mod interpreter;
mod tensor;
mod error;
mod type_checker;

use lexer::Lexer;
use parser::Parser;
use interpreter::Interpreter;
use environment::Environment;
use object::Object;
use error::FluxError;
use miette::Report;
use pyo3::types::PyAnyMethods;
use std::io::{self, Write};
use std::env;
use std::fs;
use std::rc::Rc;
use std::collections::HashSet;
use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use crate::tensor::Tensor;
use rand::Rng;

fn register_builtins(env: Rc<RefCell<Environment>>) {
    let tensor_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("tensor() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(size) => {
                let s = *size as usize;
                let data = vec![1.0; s];
                let shape = vec![s];
                let t = crate::tensor::Tensor::new(data, shape)?;
                Ok(Object::Tensor(t))
            },
            Object::List(l) => {
                let mut data = Vec::new();
                for item in l.borrow().iter() {
                    match item {
                        Object::Integer(i) => data.push(*i as f64),
                        Object::Float(f) => data.push(*f),
                        _ => return Err(format!("tensor() list elements must be numbers, got {}", item)),
                    }
                }
                let shape = vec![data.len()];
                let t = crate::tensor::Tensor::new(data, shape)?;
                Ok(Object::Tensor(t))
            },
            _ => Err("tensor() argument must be an integer (size) or a list of numbers".to_string()),
        }
    });
    env.borrow_mut().set("tensor".to_string(), tensor_fn);
    
    let matrix_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("matrix() takes exactly 2 arguments (rows, cols)".to_string());
        }
        let rows = match &args[0] {
            Object::Integer(i) => *i as usize,
            _ => return Err("rows must be an integer".to_string()),
        };
        let cols = match &args[1] {
            Object::Integer(i) => *i as usize,
            _ => return Err("cols must be an integer".to_string()),
        };
        
        let data = vec![1.0; rows * cols];
        let shape = vec![rows, cols];
        let t = crate::tensor::Tensor::new(data, shape)?;
        Ok(Object::Tensor(t))
    });
    env.borrow_mut().set("matrix".to_string(), matrix_fn);
    
    let py_import_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("py_import() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(module_name) => {
                pyo3::Python::attach(|py| {
                    let module = pyo3::types::PyModule::import(py, module_name.as_str())
                        .map_err(|e| format!("Failed to import module: {}", e))?;
                    Ok(Object::PyObject(module.into()))
                })
            },
            _ => Err("py_import() argument must be a string".to_string()),
        }
    });
    env.borrow_mut().set("py_import".to_string(), py_import_fn);
    
    let range_fn = Object::NativeFn(|args| {
        let (start, stop, step) = match args.len() {
            1 => (0, match &args[0] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) }, 1),
            2 => (
                match &args[0] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) },
                match &args[1] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) },
                1
            ),
            3 => (
                match &args[0] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) },
                match &args[1] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) },
                match &args[2] { Object::Integer(i) => *i, _ => return Err("range() arg must be integer".to_string()) }
            ),
            _ => return Err("range() takes 1-3 arguments".to_string()),
        };

        let mut elements = Vec::new();
        let mut current = start;
        if step > 0 {
            while current < stop {
                elements.push(Object::Integer(current));
                current += step;
            }
        } else if step < 0 {
            while current > stop {
                elements.push(Object::Integer(current));
                current += step;
            }
        }
        Ok(Object::List(Rc::new(RefCell::new(elements))))
    });
    env.borrow_mut().set("range".to_string(), range_fn);
    
    let len_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("len() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => Ok(Object::Integer(l.borrow().len() as i64)),
            Object::String(s) => Ok(Object::Integer(s.len() as i64)),
            Object::Tensor(t) => {
                let shape = t.inner.shape();
                if shape.is_empty() {
                    Ok(Object::Integer(0))
                } else {
                    Ok(Object::Integer(shape[0] as i64))
                }
            },
            Object::PyObject(py_obj) => {
                pyo3::Python::attach(|py| {
                    let len = py_obj.bind(py).len()
                        .map_err(|e| format!("Python len() error: {}", e))?;
                    Ok(Object::Integer(len as i64))
                })
            },
            Object::Dictionary(d) => Ok(Object::Integer(d.borrow().len() as i64)),
            Object::Set(s) => Ok(Object::Integer(s.borrow().len() as i64)),
            _ => Err(format!("len() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("len".to_string(), len_fn);
    
    let type_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("type() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(_) => Ok(Object::String("int".to_string())),
            Object::Float(_) => Ok(Object::String("float".to_string())),
            Object::String(_) => Ok(Object::String("string".to_string())),
            Object::Boolean(_) => Ok(Object::String("bool".to_string())),
            Object::Null => Ok(Object::String("null".to_string())),
            Object::List(_) => Ok(Object::String("list".to_string())),
            Object::Tensor(_) => Ok(Object::String("tensor".to_string())),
            Object::Function { .. } => Ok(Object::String("function".to_string())),
            Object::NativeFn(_) => Ok(Object::String("native_function".to_string())),
            Object::PyObject(_) => Ok(Object::String("pyobject".to_string())),
            Object::ReturnValue(_) => Ok(Object::String("return_value".to_string())),
            Object::Break => Ok(Object::String("break".to_string())),
            Object::Continue => Ok(Object::String("continue".to_string())),
            Object::Dictionary(_) => Ok(Object::String("dict".to_string())),
            Object::Set(_) => Ok(Object::String("set".to_string())),
            Object::Module { .. } => Ok(Object::String("module".to_string())),
            Object::Slice { .. } => Ok(Object::String("slice".to_string())),
        }
    });
    let repr_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("repr() takes exactly 1 argument".to_string()); }
        Ok(Object::String(args[0].repr()))
    });
    env.borrow_mut().set("repr".to_string(), repr_fn);

    let divmod_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("divmod() takes exactly 2 arguments".to_string()); }
        match (&args[0], &args[1]) {
            (Object::Integer(a), Object::Integer(b)) => {
                if *b == 0 { return Err("divmod() by zero".to_string()); }
                let q = a / b;
                let r = a % b;
                Ok(Object::List(Rc::new(RefCell::new(vec![Object::Integer(q), Object::Integer(r)]))))
            },
            _ => Err(format!("divmod() not supported for {} and {}", args[0], args[1])),
        }
    });
    env.borrow_mut().set("divmod".to_string(), divmod_fn);

    env.borrow_mut().set("type".to_string(), type_fn);

    let zeros_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("zeros() takes exactly 1 argument (shape)".to_string()); }
        let shape = match &args[0] {
            Object::List(l) => l.borrow().iter().map(|item| match item {
                Object::Integer(i) => *i as usize,
                _ => 0
            }).collect::<Vec<usize>>(),
            _ => return Err("zeros() argument must be a list of integers".to_string()),
        };
        Ok(Object::Tensor(crate::tensor::Tensor::zeros(shape)))
    });
    env.borrow_mut().set("zeros".to_string(), zeros_fn);

    let ones_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("ones() takes exactly 1 argument (shape)".to_string()); }
        let shape = match &args[0] {
            Object::List(l) => l.borrow().iter().map(|item| match item {
                Object::Integer(i) => *i as usize,
                _ => 0
            }).collect::<Vec<usize>>(),
            _ => return Err("ones() argument must be a list of integers".to_string()),
        };
        Ok(Object::Tensor(crate::tensor::Tensor::ones(shape)))
    });
    env.borrow_mut().set("ones".to_string(), ones_fn);

    let rand_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("rand() takes exactly 1 argument (shape)".to_string()); }
        let shape = match &args[0] {
            Object::List(l) => l.borrow().iter().map(|item| match item {
                Object::Integer(i) => *i as usize,
                _ => 0
            }).collect::<Vec<usize>>(),
            _ => return Err("rand() argument must be a list of integers".to_string()),
        };
        Ok(Object::Tensor(crate::tensor::Tensor::rand(shape)))
    });
    env.borrow_mut().set("rand".to_string(), rand_fn);

    let reshape_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("reshape() takes exactly 2 arguments (tensor, shape)".to_string()); }
        let t = match &args[0] {
            Object::Tensor(t) => t,
            _ => return Err("reshape() first argument must be a tensor".to_string()),
        };
        let shape = match &args[1] {
            Object::List(l) => l.borrow().iter().map(|item| match item {
                Object::Integer(i) => *i as usize,
                _ => 0
            }).collect::<Vec<usize>>(),
            _ => return Err("reshape() second argument must be a list of integers".to_string()),
        };
        t.reshape(shape).map(Object::Tensor)
    });
    env.borrow_mut().set("reshape".to_string(), reshape_fn);



    let dot_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("dot() takes exactly 2 arguments".to_string()); }
        let a = match &args[0] { Object::Tensor(t) => t, _ => return Err("dot() args must be tensors".to_string()) };
        let b = match &args[1] { Object::Tensor(t) => t, _ => return Err("dot() args must be tensors".to_string()) };
        a.matmul(b).map(Object::Tensor)
    });
    env.borrow_mut().set("dot".to_string(), dot_fn);


    let abs_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("abs() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(i.abs())),
            Object::Float(f) => Ok(Object::Float(f.abs())),
            Object::Tensor(t) => Ok(Object::Tensor(t.abs())),
            _ => Err(format!("abs() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("abs".to_string(), abs_fn);

    let sum_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("sum() takes 1 or 2 arguments (iterable, [start])".to_string());
        }
        
        let mut total_int = 0;
        let mut total_float = 0.0;
        let mut has_float = false;

        if args.len() == 2 {
            match &args[1] {
                Object::Integer(i) => total_int = *i,
                Object::Float(f) => {
                    has_float = true;
                    total_float = *f;
                },
                _ => return Err(format!("sum() start value must be numeric, got {}", args[1])),
            }
        }

        match &args[0] {
            Object::Tensor(t) => {
                let s = t.sum();
                if has_float { Ok(Object::Float(total_float + s)) }
                else { Ok(Object::Float(total_int as f64 + s)) }
            },
            Object::List(l) => {
                if has_float { total_float += total_int as f64; }

                for item in l.borrow().iter() {
                    match item {
                        Object::Integer(i) => {
                            if has_float {
                                total_float += *i as f64;
                            } else {
                                total_int += i;
                            }
                        },
                        Object::Float(f) => {
                            if !has_float {
                                has_float = true;
                                total_float = total_int as f64;
                            }
                            total_float += f;
                        },
                        _ => return Err(format!("sum() encountered non-numeric element: {}", item)),
                    }
                }

                if has_float {
                    Ok(Object::Float(total_float))
                } else {
                    Ok(Object::Integer(total_int))
                }
            },
            _ => Err(format!("sum() first argument must be a list or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("sum".to_string(), sum_fn);

    let fsum_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("fsum() takes exactly 1 argument (iterable)".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut total: f64 = 0.0;
                for item in l.borrow().iter() {
                    match item {
                        Object::Integer(i) => total += *i as f64,
                        Object::Float(f) => total += *f,
                        _ => return Err(format!("fsum() elements must be numeric, got {}", item)),
                    }
                }
                Ok(Object::Float(total))
            },
            Object::Set(set_obj) => {
                let mut total: f64 = 0.0;
                for item in set_obj.borrow().iter() {
                    match item {
                        Object::Integer(i) => total += *i as f64,
                        Object::Float(f) => total += *f,
                        _ => return Err(format!("fsum() elements must be numeric, got {}", item)),
                    }
                }
                Ok(Object::Float(total))
            },
            Object::Tensor(t) => Ok(Object::Float(t.sum())),
            _ => Err(format!("fsum() argument must be an iterable, got {}", args[0])),
        }
    });
    env.borrow_mut().set("fsum".to_string(), fsum_fn);

    let mean_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("mean() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Tensor(t) => Ok(t.mean().map(Object::Float).unwrap_or(Object::Null)),
            Object::List(l) => {
                let list_borrow = l.borrow();
                if list_borrow.is_empty() { return Ok(Object::Null); }
                let mut total = 0.0;
                for item in list_borrow.iter() {
                    match item {
                        Object::Integer(i) => total += *i as f64,
                        Object::Float(f) => total += f,
                        _ => return Err(format!("mean() encountered non-numeric element: {}", item)),
                    }
                }
                Ok(Object::Float(total / list_borrow.len() as f64))
            },
            _ => Err(format!("mean() argument must be a list or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("mean".to_string(), mean_fn);

    let std_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("std() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Float(t.std())),
            _ => Err(format!("std() argument must be a tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("std".to_string(), std_fn);

    let var_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("var() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Float(t.var())),
            _ => Err(format!("var() argument must be a tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("var".to_string(), var_fn);

    let min_fn = Object::NativeFn(|args| {
        if args.len() == 0 {
            return Err("min() expects at least 1 argument".to_string());
        }
        if args.len() == 1 {
            if let Object::Tensor(t) = &args[0] {
                return Ok(Object::Float(t.min()));
            }
        }

        let elements = if args.len() == 1 {
            match &args[0] {
                Object::List(l) => l.borrow().clone(),
                _ => args.clone(),
            }
        } else {
            args.clone()
        };

        if elements.is_empty() { return Err("min() arg is an empty sequence".to_string()); }
        let mut all_int = true;
        let mut all_str = false;
        let mut min_i = 0;
        let mut min_f = 0.0;
        let mut min_s = String::new();
        let mut first = true;
        for item in elements {
            match item {
                Object::Integer(i) => {
                    if first {
                        min_i = i;
                    } else if all_str {
                        return Err("min() cannot compare strings and integers".to_string());
                    } else {
                        if (i as f64) < (if all_int { min_i as f64 } else { min_f }) {
                            min_i = i;
                            if !all_int { min_f = i as f64; }
                        }
                    }
                },
                Object::Float(f) => {
                    if first {
                        all_int = false;
                        min_f = f;
                    } else if all_str {
                        return Err("min() cannot compare strings and floats".to_string());
                    } else {
                        if all_int {
                            all_int = false;
                            min_f = min_i as f64;
                        }
                        if f < min_f {
                            min_f = f;
                        }
                    }
                },
                Object::String(s) => {
                    if first {
                        all_int = false;
                        all_str = true;
                        min_s = s.clone();
                    } else if !all_str {
                        return Err("min() cannot compare numbers and strings".to_string());
                    } else {
                        if s < min_s {
                            min_s = s.clone();
                        }
                    }
                }
                _ => return Err(format!("min() encountered non-comparable element: {}", item)),
            }
            first = false;
        }
        if all_str { Ok(Object::String(min_s)) }
        else if all_int { Ok(Object::Integer(min_i)) } 
        else { Ok(Object::Float(min_f)) }
    });
    env.borrow_mut().set("min".to_string(), min_fn);

    let max_fn = Object::NativeFn(|args| {
        if args.len() == 0 {
            return Err("max() expects at least 1 argument".to_string());
        }
        if args.len() == 1 {
            if let Object::Tensor(t) = &args[0] {
                return Ok(Object::Float(t.max()));
            }
        }

        let elements = if args.len() == 1 {
            match &args[0] {
                Object::List(l) => l.borrow().clone(),
                _ => args.clone(),
            }
        } else {
            args.clone()
        };

        if elements.is_empty() { return Err("max() arg is an empty sequence".to_string()); }
        let mut all_int = true;
        let mut all_str = false;
        let mut max_i = 0;
        let mut max_f = 0.0;
        let mut max_s = String::new();
        let mut first = true;
        for item in elements {
            match item {
                Object::Integer(i) => {
                    if first {
                        max_i = i;
                    } else if all_str {
                        return Err("max() cannot compare strings and integers".to_string());
                    } else {
                        if (i as f64) > (if all_int { max_i as f64 } else { max_f }) {
                            max_i = i;
                            if !all_int { max_f = i as f64; }
                        }
                    }
                },
                Object::Float(f) => {
                    if first {
                        all_int = false;
                        max_f = f;
                    } else if all_str {
                        return Err("max() cannot compare strings and floats".to_string());
                    } else {
                        if all_int {
                            all_int = false;
                            max_f = max_i as f64;
                        }
                        if f > max_f {
                            max_f = f;
                        }
                    }
                },
                Object::String(s) => {
                    if first {
                        all_int = false;
                        all_str = true;
                        max_s = s.clone();
                    } else if !all_str {
                        return Err("max() cannot compare numbers and strings".to_string());
                    } else {
                        if s > max_s {
                            max_s = s.clone();
                        }
                    }
                }
                _ => return Err(format!("max() encountered non-comparable element: {}", item)),
            }
            first = false;
        }
        if all_str { Ok(Object::String(max_s)) }
        else if all_int { Ok(Object::Integer(max_i)) } 
        else { Ok(Object::Float(max_f)) }
    });
    env.borrow_mut().set("max".to_string(), max_fn);


    let all_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("all() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                for item in l.borrow().iter() {
                    if !item.is_truthy() {
                        return Ok(Object::Boolean(false));
                    }
                }
                Ok(Object::Boolean(true))
            },
            Object::String(s) => {
                for c in s.chars() {
                    // Chaque caractère est une chaîne non vide, donc toujours truthy.
                    // Mais on suit la sémantique Python d'itération.
                    if !Object::String(c.to_string()).is_truthy() {
                        return Ok(Object::Boolean(false));
                    }
                }
                Ok(Object::Boolean(true))
            },
            Object::Tensor(t) => Ok(Object::Boolean(t.all())),
            _ => Err(format!("all() argument must be a list, string, or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("all".to_string(), all_fn);

    let any_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("any() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                for item in l.borrow().iter() {
                    if item.is_truthy() {
                        return Ok(Object::Boolean(true));
                    }
                }
                Ok(Object::Boolean(false))
            },
            Object::String(s) => {
                for c in s.chars() {
                    if Object::String(c.to_string()).is_truthy() {
                        return Ok(Object::Boolean(true));
                    }
                }
                Ok(Object::Boolean(false))
            },
            Object::Tensor(t) => Ok(Object::Boolean(t.any())),
            _ => Err(format!("any() argument must be a list, string, or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("any".to_string(), any_fn);

    let prod_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("prod() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Float(t.prod())),
            Object::List(l) => {
                let mut total_int = 1;
                let mut total_float = 1.0;
                let mut has_float = false;
                for item in l.borrow().iter() {
                    match item {
                        Object::Integer(i) => {
                            if has_float { total_float *= *i as f64; }
                            else { total_int *= i; }
                        },
                        Object::Float(f) => {
                            if !has_float {
                                has_float = true;
                                total_float = total_int as f64;
                            }
                            total_float *= f;
                        },
                        _ => return Err(format!("prod() encountered non-numeric element: {}", item)),
                    }
                }
                if has_float { Ok(Object::Float(total_float)) }
                else { Ok(Object::Integer(total_int)) }
            },
            _ => Err(format!("prod() argument must be a list or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("prod".to_string(), prod_fn);

    let reverse_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("reverse() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut rev_l = l.borrow().clone();
                rev_l.reverse();
                Ok(Object::List(Rc::new(RefCell::new(rev_l))))
            },
            Object::String(s) => {
                let rev_s: String = s.chars().rev().collect();
                Ok(Object::String(rev_s))
            },
            _ => Err(format!("reverse() argument must be a list or string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("reverse".to_string(), reverse_fn);

    let sorted_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sorted() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut sorted_l = l.borrow().clone();
                let mut err = None;
                sorted_l.sort_by(|a, b| {
                    match (a, b) {
                        (Object::Integer(ai), Object::Integer(bi)) => ai.cmp(bi),
                        (Object::Float(af), Object::Float(bf)) => af.partial_cmp(bf).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::Integer(ai), Object::Float(bf)) => (*ai as f64).partial_cmp(bf).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::Float(af), Object::Integer(bi)) => af.partial_cmp(&(*bi as f64)).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::String(as_str), Object::String(bs_str)) => as_str.cmp(bs_str),
                        _ => {
                            err = Some(format!("Cannot compare {} and {}", a, b));
                            std::cmp::Ordering::Equal
                        }
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                Ok(Object::List(Rc::new(RefCell::new(sorted_l))))
            },
            _ => Err(format!("sorted() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("sorted".to_string(), sorted_fn);

    let sort_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sort() takes exactly 1 argument (list)".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut err = None;
                l.borrow_mut().sort_by(|a, b| {
                    match (a, b) {
                        (Object::Integer(ai), Object::Integer(bi)) => ai.cmp(bi),
                        (Object::Float(af), Object::Float(bf)) => af.partial_cmp(bf).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::Integer(ai), Object::Float(bf)) => (*ai as f64).partial_cmp(bf).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::Float(af), Object::Integer(bi)) => af.partial_cmp(&(*bi as f64)).unwrap_or(std::cmp::Ordering::Equal),
                        (Object::String(as_str), Object::String(bs_str)) => as_str.cmp(bs_str),
                        _ => {
                            err = Some(format!("Cannot compare {} and {}", a, b));
                            std::cmp::Ordering::Equal
                        }
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                Ok(args[0].clone())
            },
            _ => Err(format!("sort() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("sort".to_string(), sort_fn);

    let round_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("round() takes 1 or 2 arguments".to_string());
        }
        if let Object::Tensor(t) = &args[0] {
            return Ok(Object::Tensor(t.round()));
        }
        let num = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("round() first argument must be numeric, got {}", args[0])),
        };
        let ndigits = if args.len() == 2 {
            match &args[1] {
                Object::Integer(i) => *i as i32,
                _ => return Err(format!("round() ndigits must be an integer, got {}", args[1])),
            }
        } else {
            0
        };

        let factor = 10.0f64.powi(ndigits);
        let rounded = (num * factor).round() / factor;
        
        if ndigits <= 0 {
            Ok(Object::Integer(rounded as i64))
        } else {
            Ok(Object::Float(rounded))
        }
    });
    env.borrow_mut().set("round".to_string(), round_fn);


    let upper_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("upper() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.to_uppercase())),
            _ => Err(format!("upper() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("upper".to_string(), upper_fn);

    let lower_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("lower() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.to_lowercase())),
            _ => Err(format!("lower() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("lower".to_string(), lower_fn);

    let strip_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("strip() takes 1 or 2 arguments".to_string());
        }
        let s = match &args[0] {
            Object::String(s) => s,
            _ => return Err(format!("strip() first argument must be a string, got {}", args[0])),
        };
        if args.len() == 1 {
            Ok(Object::String(s.trim().to_string()))
        } else {
            let chars = match &args[1] {
                Object::String(c) => c,
                _ => return Err(format!("strip() second argument must be a string, got {}", args[1])),
            };
            let chars_vec: Vec<char> = chars.chars().collect();
            Ok(Object::String(s.trim_matches(|c| chars_vec.contains(&c)).to_string()))
        }
    });
    env.borrow_mut().set("strip".to_string(), strip_fn);


    let int_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("int() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Integer(*f as i64)),
            Object::String(s) => s.parse::<i64>().map(Object::Integer).map_err(|e| format!("int() parse error: {}", e)),
            Object::Boolean(b) => Ok(Object::Integer(if *b { 1 } else { 0 })),
            _ => Err(format!("int() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("int".to_string(), int_fn);

    let float_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("float() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Float(f) => Ok(Object::Float(*f)),
            Object::Integer(i) => Ok(Object::Float(*i as f64)),
            Object::String(s) => s.parse::<f64>().map(Object::Float).map_err(|e| format!("float() parse error: {}", e)),
            Object::Boolean(b) => Ok(Object::Float(if *b { 1.0 } else { 0.0 })),
            _ => Err(format!("float() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("float".to_string(), float_fn);

    let str_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("str() takes exactly 1 argument".to_string());
        }
        Ok(Object::String(args[0].to_string()))
    });
    env.borrow_mut().set("str".to_string(), str_fn);

    let split_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 3 {
            return Err("split() takes 1 to 3 arguments (string, [sep], [maxsplit])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("split() first argument must be a string, got {}", args[0])),
        };
        let sep = if args.len() >= 2 {
            match &args[1] {
                Object::String(val) => Some(val.as_str()),
                Object::Null => None,
                _ => return Err(format!("split() separator must be a string or null, got {}", args[1])),
            }
        } else {
            None
        };
        let maxsplit = if args.len() == 3 {
            match &args[2] {
                Object::Integer(i) => *i as isize,
                _ => return Err(format!("split() maxsplit must be an integer, got {}", args[2])),
            }
        } else {
            -1
        };

        let items: Vec<Object> = match sep {
            None => {
                if maxsplit < 0 {
                    s.split_whitespace().map(|item| Object::String(item.to_string())).collect()
                } else {
                    s.split_whitespace().take(maxsplit as usize + 1).map(|item| Object::String(item.to_string())).collect()
                    // Rust's split_whitespace doesn't have a direct maxsplit equivalent that keeps the rest.
                    // Let's implement it manually or use a different approach.
                }
            }
            Some(sep_str) => {
                if sep_str.is_empty() {
                    s.chars().map(|c| Object::String(c.to_string())).collect()
                } else {
                    if maxsplit < 0 {
                        s.split(sep_str).map(|item| Object::String(item.to_string())).collect()
                    } else {
                        let parts: Vec<Object> = s.splitn(maxsplit as usize + 1, sep_str)
                            .map(|item| Object::String(item.to_string()))
                            .collect();
                        parts
                    }
                }
            }
        };

        // For split_whitespace with maxsplit, we need a custom implementation if we want to keep the remainder.
        let final_items = if sep.is_none() && maxsplit >= 0 {
            let mut res = Vec::new();
            let mut count = 0;
            let mut last_idx = 0;
            let mut in_whitespace = true;
            for (i, c) in s.char_indices() {
                if c.is_whitespace() {
                    if !in_whitespace {
                        if count < maxsplit {
                            res.push(Object::String(s[last_idx..i].to_string()));
                            count += 1;
                            in_whitespace = true;
                        }
                    }
                } else {
                    if in_whitespace {
                        last_idx = i;
                        in_whitespace = false;
                    }
                }
            }
            if !in_whitespace {
                res.push(Object::String(s[last_idx..].to_string()));
            }
            res
        } else {
            items
        };

        Ok(Object::List(Rc::new(RefCell::new(final_items))))
    });
    env.borrow_mut().set("split".to_string(), split_fn);

    let rsplit_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 3 {
            return Err("rsplit() takes 1 to 3 arguments (string, [sep], [maxsplit])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("rsplit() first argument must be a string, got {}", args[0])),
        };
        let sep = if args.len() >= 2 {
            match &args[1] {
                Object::String(val) => Some(val.as_str()),
                Object::Null => None,
                _ => return Err(format!("rsplit() separator must be a string or null, got {}", args[1])),
            }
        } else {
            None
        };
        let maxsplit = if args.len() == 3 {
            match &args[2] {
                Object::Integer(i) => *i as isize,
                _ => return Err(format!("rsplit() maxsplit must be an integer, got {}", args[2])),
            }
        } else {
            -1
        };

        let items: Vec<Object> = match sep {
            None => {
                // Simplified whitespace rsplit
                if maxsplit < 0 {
                    s.split_whitespace().map(|item| Object::String(item.to_string())).collect()
                } else {
                    // Manual rsplit for whitespace
                    let mut res = Vec::new();
                    let mut count = 0;
                    let mut last_idx = s.len();
                    let mut in_whitespace = true;
                    for (i, c) in s.char_indices().rev() {
                        if c.is_whitespace() {
                            if !in_whitespace {
                                if count < maxsplit {
                                    res.push(Object::String(s[i + c.len_utf8()..last_idx].to_string()));
                                    count += 1;
                                    in_whitespace = true;
                                }
                            }
                        } else {
                            if in_whitespace {
                                last_idx = i + c.len_utf8();
                                in_whitespace = false;
                            }
                        }
                    }
                    if !in_whitespace {
                        res.push(Object::String(s[0..last_idx].to_string()));
                    }
                    res.reverse();
                    res
                }
            }
            Some(sep_str) => {
                if sep_str.is_empty() {
                    s.chars().map(|c| Object::String(c.to_string())).collect()
                } else {
                    if maxsplit < 0 {
                        s.split(sep_str).map(|item| Object::String(item.to_string())).collect()
                    } else {
                        let mut res: Vec<Object> = s.rsplitn(maxsplit as usize + 1, sep_str)
                            .map(|item| Object::String(item.to_string()))
                            .collect();
                        res.reverse();
                        res
                    }
                }
            }
        };
        Ok(Object::List(Rc::new(RefCell::new(items))))
    });
    env.borrow_mut().set("rsplit".to_string(), rsplit_fn);

    let join_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("join() takes exactly 2 arguments (iterable, separator)".to_string());
        }
        let sep = match &args[1] {
            Object::String(s) => s,
            _ => return Err(format!("join() second argument must be a string, got {}", args[1])),
        };
        match &args[0] {
            Object::List(l) => {
                let result = l.borrow().iter().map(|obj| obj.to_string()).collect::<Vec<String>>().join(sep);
                Ok(Object::String(result))
            },
            Object::Set(s) => {
                let result = s.borrow().iter().map(|obj| obj.to_string()).collect::<Vec<String>>().join(sep);
                Ok(Object::String(result))
            },
            _ => Err(format!("join() first argument must be a list or set, got {}", args[0])),
        }
    });
    env.borrow_mut().set("join".to_string(), join_fn);

    let replace_fn = Object::NativeFn(|args| {
        if args.len() < 3 || args.len() > 4 {
            return Err("replace() takes 3 or 4 arguments (string, old, new, [count])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("replace() first argument must be a string, got {}", args[0])),
        };
        let old = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("replace() second argument must be a string, got {}", args[1])),
        };
        let new = match &args[2] {
            Object::String(val) => val,
            _ => return Err(format!("replace() third argument must be a string, got {}", args[2])),
        };
        let count = if args.len() == 4 {
            match &args[3] {
                Object::Integer(i) => *i as usize,
                _ => return Err(format!("replace() count must be an integer, got {}", args[3])),
            }
        } else {
            0 // 0 means all for our custom logic if we implement it, or we can use usize::MAX
        };

        if args.len() == 3 {
            Ok(Object::String(s.replace(old, new)))
        } else {
            Ok(Object::String(s.replacen(old, new, count)))
        }
    });
    env.borrow_mut().set("replace".to_string(), replace_fn);

    let find_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 {
            return Err("find() takes 2 to 4 arguments (substring, [start], [end])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("find() first argument must be a string, got {}", args[0])),
        };
        let sub = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("find() second argument must be a string, got {}", args[1])),
        };
        let len = s.chars().count();
        let start = if args.len() >= 3 {
            match &args[2] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("find() start must be an integer".to_string()),
            }
        } else { 0 };
        let end = if args.len() == 4 {
            match &args[3] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("find() end must be an integer".to_string()),
            }
        } else { len };

        if start < end && start < len {
            let sliced: String = s.chars().skip(start).take(end - start).collect();
            match sliced.find(sub) {
                Some(idx) => {
                    let char_offset = sliced[..idx].chars().count();
                    return Ok(Object::Integer((char_offset + start) as i64));
                }
                None => {}
            }
        }
        Ok(Object::Integer(-1))
    });
    env.borrow_mut().set("find".to_string(), find_fn);

    let floor_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("floor() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Integer(f.floor() as i64)),
            Object::Tensor(t) => Ok(Object::Tensor(t.floor())),
            _ => Err(format!("floor() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("floor".to_string(), floor_fn);

    let ceil_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("ceil() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Integer(f.ceil() as i64)),
            Object::Tensor(t) => Ok(Object::Tensor(t.ceil())),
            _ => Err(format!("ceil() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("ceil".to_string(), ceil_fn);

    let asin_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("asin() takes exactly 1 argument".to_string()); }
        let val = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("asin() arg must be numeric".to_string()) };
        Ok(Object::Float(val.asin()))
    });
    env.borrow_mut().set("asin".to_string(), asin_fn);

    let acos_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("acos() takes exactly 1 argument".to_string()); }
        let val = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("acos() arg must be numeric".to_string()) };
        Ok(Object::Float(val.acos()))
    });
    env.borrow_mut().set("acos".to_string(), acos_fn);

    let atan_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("atan() takes exactly 1 argument".to_string()); }
        let val = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("atan() arg must be numeric".to_string()) };
        Ok(Object::Float(val.atan()))
    });
    env.borrow_mut().set("atan".to_string(), atan_fn);

    let atan2_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("atan2() takes exactly 2 arguments (y, x)".to_string()); }
        let y = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("atan2() y must be numeric".to_string()) };
        let x = match &args[1] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("atan2() x must be numeric".to_string()) };
        Ok(Object::Float(y.atan2(x)))
    });
    env.borrow_mut().set("atan2".to_string(), atan2_fn);

    let sinh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("sinh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).sinh())),
            Object::Float(f) => Ok(Object::Float(f.sinh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.sinh()) })),
            _ => Err("sinh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("sinh".to_string(), sinh_fn);

    let cosh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("cosh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).cosh())),
            Object::Float(f) => Ok(Object::Float(f.cosh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.cosh()) })),
            _ => Err("cosh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("cosh".to_string(), cosh_fn);

    let tanh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("tanh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).tanh())),
            Object::Float(f) => Ok(Object::Float(f.tanh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.tanh()) })),
            _ => Err("tanh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("tanh".to_string(), tanh_fn);

    let asinh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("asinh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).asinh())),
            Object::Float(f) => Ok(Object::Float(f.asinh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.asinh()) })),
            _ => Err("asinh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("asinh".to_string(), asinh_fn);

    let acosh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("acosh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).acosh())),
            Object::Float(f) => Ok(Object::Float(f.acosh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.acosh()) })),
            _ => Err("acosh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("acosh".to_string(), acosh_fn);

    let atanh_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("atanh() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).atanh())),
            Object::Float(f) => Ok(Object::Float(f.atanh())),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| x.atanh()) })),
            _ => Err("atanh() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("atanh".to_string(), atanh_fn);

    let erf_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("erf() takes exactly 1 argument".to_string()); }
        use statrs::function::erf;
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float(erf::erf(*i as f64))),
            Object::Float(f) => Ok(Object::Float(erf::erf(*f))),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| erf::erf(x)) })),
            _ => Err("erf() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("erf".to_string(), erf_fn);

    let erfc_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("erfc() takes exactly 1 argument".to_string()); }
        use statrs::function::erf;
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float(erf::erfc(*i as f64))),
            Object::Float(f) => Ok(Object::Float(erf::erfc(*f))),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| erf::erfc(x)) })),
            _ => Err("erfc() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("erfc".to_string(), erfc_fn);

    let gamma_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("gamma() takes exactly 1 argument".to_string()); }
        use statrs::function::gamma;
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float(gamma::gamma(*i as f64))),
            Object::Float(f) => Ok(Object::Float(gamma::gamma(*f))),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| gamma::gamma(x)) })),
            _ => Err("gamma() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("gamma".to_string(), gamma_fn);

    let lgamma_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("lgamma() takes exactly 1 argument".to_string()); }
        use statrs::function::gamma;
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float(gamma::ln_gamma(*i as f64))),
            Object::Float(f) => Ok(Object::Float(gamma::ln_gamma(*f))),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(|x| gamma::ln_gamma(x)) })),
            _ => Err("lgamma() arg must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("lgamma".to_string(), lgamma_fn);

    let fmod_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("fmod() takes exactly 2 arguments (x, y)".to_string()); }
        let x = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("fmod() x must be numeric".to_string()) };
        let y = match &args[1] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("fmod() y must be numeric".to_string()) };
        Ok(Object::Float(x % y))
    });
    env.borrow_mut().set("fmod".to_string(), fmod_fn);



    let sqrt_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sqrt() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).sqrt())),
            Object::Float(f) => Ok(Object::Float(f.sqrt())),
            Object::Tensor(t) => Ok(Object::Tensor(t.sqrt())),
            _ => Err(format!("sqrt() argument must be numeric or tensor, got {}", args[0])),
        }?;
        Ok(val)
    });
    env.borrow_mut().set("sqrt".to_string(), sqrt_fn);

    let log2_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("log2() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).log2())),
            Object::Float(f) => Ok(Object::Float(f.log2())),
            Object::Tensor(t) => Ok(Object::Tensor(t.log2())),
            _ => Err(format!("log2() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("log2".to_string(), log2_fn);

    let log10_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("log10() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).log10())),
            Object::Float(f) => Ok(Object::Float(f.log10())),
            Object::Tensor(t) => Ok(Object::Tensor(t.log10())),
            _ => Err(format!("log10() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("log10".to_string(), log10_fn);

    let isinf_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("isinf() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Float(f) => Ok(Object::Boolean(f.is_infinite())),
            Object::Integer(_) => Ok(Object::Boolean(false)),
            Object::Tensor(t) => Ok(Object::Tensor(t.isinf())),
            _ => Err(format!("isinf() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isinf".to_string(), isinf_fn);

    let isnan_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("isnan() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Float(f) => Ok(Object::Boolean(f.is_nan())),
            Object::Integer(_) => Ok(Object::Boolean(false)),
            Object::Tensor(t) => Ok(Object::Tensor(t.isnan())),
            _ => Err(format!("isnan() argument must be numeric or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isnan".to_string(), isnan_fn);

    let isfinite_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("isfinite() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Float(f) => Ok(Object::Boolean(f.is_finite())),
            Object::Integer(_) => Ok(Object::Boolean(true)),
            _ => Err(format!("isfinite() argument must be numeric, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isfinite".to_string(), isfinite_fn);

    let transpose_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("transpose() takes exactly 1 argument (tensor)".to_string()); }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Tensor(t.transpose())),
            _ => Err(format!("transpose() argument must be a tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("transpose".to_string(), transpose_fn);

    let argmax_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("argmax() takes 1 or 2 arguments (tensor, [axis])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("argmax() arg 1 must be tensor".to_string()) };
        let axis = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => Some(*i as usize), _ => return Err("argmax() axis must be int".to_string()) }
        } else { None };
        Ok(t.argmax(axis))
    });
    env.borrow_mut().set("argmax".to_string(), argmax_fn);

    let argmin_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("argmin() takes 1 or 2 arguments (tensor, [axis])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("argmin() arg 1 must be tensor".to_string()) };
        let axis = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => Some(*i as usize), _ => return Err("argmin() axis must be int".to_string()) }
        } else { None };
        Ok(t.argmin(axis))
    });
    env.borrow_mut().set("argmin".to_string(), argmin_fn);

    let squeeze_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("squeeze() takes 1 or 2 arguments (tensor, [axis])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("squeeze() arg 1 must be tensor".to_string()) };
        let axis = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => Some(*i as usize), _ => return Err("squeeze() axis must be int".to_string()) }
        } else { None };
        Ok(Object::Tensor(t.squeeze(axis)?))
    });
    env.borrow_mut().set("squeeze".to_string(), squeeze_fn);

    let unsqueeze_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("unsqueeze() takes exactly 2 arguments (tensor, axis)".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("unsqueeze() arg 1 must be tensor".to_string()) };
        let axis = match &args[1] { Object::Integer(i) => *i as usize, _ => return Err("unsqueeze() axis must be int".to_string()) };
        Ok(Object::Tensor(t.unsqueeze(axis)?))
    });
    env.borrow_mut().set("unsqueeze".to_string(), unsqueeze_fn);

    let eye_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("eye() takes 1 or 2 arguments (n, [m])".to_string()); }
        let n = match &args[0] { Object::Integer(i) => *i as usize, _ => return Err("eye() n must be int".to_string()) };
        let m = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => *i as usize, _ => return Err("eye() m must be int".to_string()) }
        } else { n };
        
        let mut data = vec![0.0; n * m];
        for i in 0..std::cmp::min(n, m) {
            data[i * m + i] = 1.0;
        }
        Ok(Object::Tensor(Tensor::new(data, vec![n, m])?))
    });
    env.borrow_mut().set("eye".to_string(), eye_fn);

    let diag_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("diag() takes 1 or 2 arguments (tensor, [k])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("diag() arg 1 must be tensor".to_string()) };
        let k = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => *i as i32, _ => return Err("diag() k must be int".to_string()) }
        } else { 0 };
        Ok(Object::Tensor(t.diag(k)?))
    });
    env.borrow_mut().set("diag".to_string(), diag_fn);

    let trace_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("trace() takes exactly 1 argument (tensor)".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("trace() arg must be tensor".to_string()) };
        Ok(Object::Float(t.trace()?))
    });
    env.borrow_mut().set("trace".to_string(), trace_fn);

    let pow_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("pow() takes exactly 2 arguments (base, exp)".to_string());
        }
        let base = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("pow() base must be numeric, got {}", args[0])),
        };
        let exp = match &args[1] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("pow() exponent must be numeric, got {}", args[1])),
        };
        Ok(Object::Float(base.powf(exp)))
    });
    env.borrow_mut().set("pow".to_string(), pow_fn);

    let exp_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("exp() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => Ok(Object::Float((*i as f64).exp())),
            Object::Float(f) => Ok(Object::Float(f.exp())),
            Object::Tensor(t) => Ok(Object::Tensor(t.exp())),
            _ => Err(format!("exp() argument must be numeric or tensor, got {}", args[0])),
        }?;
        Ok(val)
    });
    env.borrow_mut().set("exp".to_string(), exp_fn);

    let log_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("log() takes 1 or 2 arguments (x, [base])".to_string());
        }
        if args.len() == 1 {
            match &args[0] {
                Object::Integer(i) => return Ok(Object::Float((*i as f64).ln())),
                Object::Float(f) => return Ok(Object::Float(f.ln())),
                Object::Tensor(t) => return Ok(Object::Tensor(t.log())),
                _ => return Err(format!("log() argument must be numeric or tensor, got {}", args[0])),
            }
        }
        let x = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("log() first argument must be numeric, got {}", args[0])),
        };
        let base = match &args[1] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("log() base must be numeric, got {}", args[1])),
        };
        Ok(Object::Float(x.log(base)))
    });
    env.borrow_mut().set("log".to_string(), log_fn);

    let startswith_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("startswith() takes exactly 2 arguments (string, prefix)".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("startswith() first argument must be a string, got {}", args[0])),
        };
        let prefix = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("startswith() second argument must be a string, got {}", args[1])),
        };
        Ok(Object::Boolean(s.starts_with(prefix)))
    });
    env.borrow_mut().set("startswith".to_string(), startswith_fn);

    let endswith_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("endswith() takes exactly 2 arguments (string, suffix)".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("endswith() first argument must be a string, got {}", args[0])),
        };
        let suffix = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("endswith() second argument must be a string, got {}", args[1])),
        };
        Ok(Object::Boolean(s.ends_with(suffix)))
    });
    env.borrow_mut().set("endswith".to_string(), endswith_fn);

    let input_fn = Object::NativeFn(|args| {
        if args.len() > 1 {
            return Err("input() takes at most 1 argument (prompt)".to_string());
        }
        if args.len() == 1 {
             match &args[0] {
                Object::String(prompt) => {
                    print!("{}", prompt);
                    io::stdout().flush().map_err(|e| format!("Flush error: {}", e))?;
                },
                _ => return Err(format!("input() prompt must be a string, got {}", args[0])),
            }
        }
        let mut buffer = String::new();
        io::stdin().read_line(&mut buffer).map_err(|e| format!("Failed to read input: {}", e))?;
        if buffer.ends_with('\n') {
            buffer.pop();
            if buffer.ends_with('\r') {
                buffer.pop();
            }
        }
        Ok(Object::String(buffer))
    });
    env.borrow_mut().set("input".to_string(), input_fn);

    let keys_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("keys() takes exactly 1 argument (dictionary)".to_string());
        }
        match &args[0] {
            Object::Dictionary(d) => {
                let keys: Vec<Object> = d.borrow().keys().cloned().collect();
                Ok(Object::List(Rc::new(RefCell::new(keys))))
            },
            _ => Err(format!("keys() argument must be a dictionary, got {}", args[0])),
        }
    });
    env.borrow_mut().set("keys".to_string(), keys_fn);

    let values_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("values() takes exactly 1 argument (dictionary)".to_string());
        }
        match &args[0] {
            Object::Dictionary(d) => {
                let values: Vec<Object> = d.borrow().values().cloned().collect();
                Ok(Object::List(Rc::new(RefCell::new(values))))
            },
            _ => Err(format!("values() argument must be a dictionary, got {}", args[0])),
        }
    });
    env.borrow_mut().set("values".to_string(), values_fn);

    let items_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("items() takes exactly 1 argument (dictionary)".to_string()); }
        match &args[0] {
            Object::Dictionary(d) => {
                let items: Vec<Object> = d.borrow().iter()
                    .map(|(k, v)| Object::List(Rc::new(RefCell::new(vec![k.clone(), v.clone()]))))
                    .collect();
                Ok(Object::List(Rc::new(RefCell::new(items))))
            },
            _ => Err(format!("items() argument must be a dictionary, got {}", args[0])),
        }
    });
    env.borrow_mut().set("items".to_string(), items_fn);

    let get_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 {
             return Err("get() takes 2 or 3 arguments (dictionary, key, [default])".to_string());
        }
        let dict = match &args[0] {
            Object::Dictionary(d) => d,
            _ => return Err(format!("get() first argument must be a dictionary, got {}", args[0])),
        };
        let default = if args.len() == 3 { args[2].clone() } else { Object::Null };
        match dict.borrow().get(&args[1]) {
            Some(v) => Ok(v.clone()),
            None => Ok(default),
        }
    });
    env.borrow_mut().set("get".to_string(), get_fn);

    let clear_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("clear() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::List(l) => l.borrow_mut().clear(),
            Object::Dictionary(d) => d.borrow_mut().clear(),
            Object::Set(s) => s.borrow_mut().clear(),
            _ => return Err(format!("clear() argument must be a list, dict, or set, got {}", args[0])),
        }
        Ok(Object::Null)
    });
    env.borrow_mut().set("clear".to_string(), clear_fn);

    let unique_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("unique() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::List(l) => {
                let mut seen = std::collections::HashSet::new();
                let mut unique_list = Vec::new();
                for item in l.borrow().iter() {
                    if seen.insert(item.clone()) {
                        unique_list.push(item.clone());
                    }
                }
                Ok(Object::List(Rc::new(RefCell::new(unique_list))))
            },
            _ => Err(format!("unique() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("unique".to_string(), unique_fn);

    let zip_fn = Object::NativeFn(|args| {
        if args.is_empty() {
            return Ok(Object::List(Rc::new(RefCell::new(Vec::new()))));
        }
        let mut lists = Vec::new();
        for arg in args {
            match arg {
                Object::List(l) => lists.push(l.borrow().clone()),
                _ => return Err(format!("zip() arguments must be lists, got {}", arg)),
            }
        }
        
        let min_len = lists.iter().map(|l| l.len()).min().unwrap_or(0);
        let mut zipped = Vec::new();
        for i in 0..min_len {
            let mut row = Vec::new();
            for list in &lists {
                row.push(list[i].clone());
            }
            zipped.push(Object::List(Rc::new(RefCell::new(row))));
        }
        Ok(Object::List(Rc::new(RefCell::new(zipped))))
    });
    env.borrow_mut().set("zip".to_string(), zip_fn);

    let enumerate_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("enumerate() takes 1 or 2 arguments (list, [start])".to_string());
        }
        let l = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("enumerate() first argument must be a list, got {}", args[0])),
        };
        let start = if args.len() == 2 {
            match &args[1] {
                Object::Integer(i) => *i,
                _ => return Err(format!("enumerate() start must be an integer, got {}", args[1])),
            }
        } else {
            0
        };
        let enumerated: Vec<Object> = l.borrow().iter().enumerate()
            .map(|(i, el)| Object::List(Rc::new(RefCell::new(vec![Object::Integer(i as i64 + start), el.clone()]))))
            .collect();
        Ok(Object::List(Rc::new(RefCell::new(enumerated))))
    });
    env.borrow_mut().set("enumerate".to_string(), enumerate_fn);

    let sin_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sin() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("sin() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(val.sin()))
    });
    env.borrow_mut().set("sin".to_string(), sin_fn);

    let cos_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("cos() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("cos() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(val.cos()))
    });
    env.borrow_mut().set("cos".to_string(), cos_fn);

    let tan_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("tan() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("tan() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(val.tan()))
    });
    env.borrow_mut().set("tan".to_string(), tan_fn);

    let isdigit_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("isdigit() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))),
            _ => Err(format!("isdigit() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isdigit".to_string(), isdigit_fn);

    let count_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 {
            return Err("count() takes 2 to 4 arguments (item, [start], [end])".to_string());
        }
        match &args[0] {
            Object::List(list) => {
                let list_borrow = list.borrow();
                let len = list_borrow.len();
                let target = &args[1];
                let start = if args.len() >= 3 {
                    match &args[2] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("count() start must be an integer".to_string()),
                    }
                } else { 0 };
                let end = if args.len() == 4 {
                    match &args[3] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("count() end must be an integer".to_string()),
                    }
                } else { len };

                if start >= end || start >= len { return Ok(Object::Integer(0)); }

                let count = list_borrow[start..end].iter().filter(|x| *x == target).count();
                Ok(Object::Integer(count as i64))
            },
            Object::String(s) => {
                let len = s.chars().count();
                let sub = match &args[1] {
                    Object::String(val) => val,
                    _ => return Err(format!("count() second argument must be a string, got {}", args[1])),
                };
                let start = if args.len() >= 3 {
                    match &args[2] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("count() start must be an integer".to_string()),
                    }
                } else { 0 };
                let end = if args.len() == 4 {
                    match &args[3] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("count() end must be an integer".to_string()),
                    }
                } else { len };

                if start >= end || start >= len { return Ok(Object::Integer(0)); }
                
                // For strings, we need to convert char indices to byte indices for slicing if we use s[start..end]
                // But char-based counting is safer.
                let sliced: String = s.chars().skip(start).take(end - start).collect();
                if sub.is_empty() {
                    return Ok(Object::Integer((sliced.chars().count() + 1) as i64));
                }
                let c = sliced.matches(sub).count();
                Ok(Object::Integer(c as i64))
            },
            _ => Err(format!("count() first argument must be a list or string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("count".to_string(), count_fn);




    let exit_fn = Object::NativeFn(|args| {
        let code = if args.len() == 1 {
            match &args[0] {
                Object::Integer(i) => *i as i32,
                _ => 0,
            }
        } else {
            0
        };
        std::process::exit(code);
    });
    env.borrow_mut().set("exit".to_string(), exit_fn);

    let random_fn = Object::NativeFn(|_| {
        let mut rng = rand::thread_rng();
        Ok(Object::Float(rng.gen()))
    });
    env.borrow_mut().set("random".to_string(), random_fn);

    let randint_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("randint() takes exactly 2 arguments (low, high)".to_string());
        }
        let low = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("randint() low must be integer, got {}", args[0])),
        };
        let high = match &args[1] {
            Object::Integer(i) => *i,
            _ => return Err(format!("randint() high must be integer, got {}", args[1])),
        };
        if low > high {
            return Err(format!("randint() low ({}) > high ({})", low, high));
        }
        let mut rng = rand::thread_rng();
        Ok(Object::Integer(rng.gen_range(low..=high)))
    });
    env.borrow_mut().set("randint".to_string(), randint_fn);

    let list_conv_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("list() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => {
                let items = s.chars().map(|c| Object::String(c.to_string())).collect();
                Ok(Object::List(Rc::new(RefCell::new(items))))
            },
            Object::List(l) => Ok(Object::List(l.clone())),
            Object::Set(s) => {
                let items: Vec<Object> = s.borrow().iter().cloned().collect();
                Ok(Object::List(Rc::new(RefCell::new(items))))
            },
            Object::Dictionary(d) => {
                let items: Vec<Object> = d.borrow().keys().cloned().collect();
                Ok(Object::List(Rc::new(RefCell::new(items))))
            },
            Object::Tensor(t) => {
                let items: Vec<Object> = t.inner.iter().map(|&x| Object::Float(x)).collect();
                Ok(Object::List(Rc::new(RefCell::new(items))))
            },
            _ => Err(format!("list() conversion not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("list".to_string(), list_conv_fn);

    let set_conv_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("set() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => {
                let mut set = HashSet::new();
                for c in s.chars() {
                    set.insert(Object::String(c.to_string()));
                }
                Ok(Object::Set(Rc::new(RefCell::new(set))))
            },
            Object::List(l) => {
                let mut set = HashSet::new();
                for item in l.borrow().iter() {
                    set.insert(item.clone());
                }
                Ok(Object::Set(Rc::new(RefCell::new(set))))
            },
            Object::Set(s) => Ok(Object::Set(s.clone())),
            Object::Dictionary(d) => {
                let mut set = HashSet::new();
                for k in d.borrow().keys() {
                    set.insert(k.clone());
                }
                Ok(Object::Set(Rc::new(RefCell::new(set))))
            },
            _ => Err(format!("set() conversion not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("set".to_string(), set_conv_fn);

    let append_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("append() takes exactly 2 arguments (list, item)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("append() first argument must be a list, got {}", args[0])),
        };
        let item = args[1].clone();
        list.borrow_mut().push(item);
        Ok(args[0].clone())  // Return the list itself
    });
    env.borrow_mut().set("append".to_string(), append_fn);

    let extend_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("extend() takes exactly 2 arguments (list1, list2)".to_string());
        }
        let list1 = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("extend() first argument must be a list, got {}", args[0])),
        };
        let list2 = match &args[1] {
            Object::List(val) => val,
            _ => return Err(format!("extend() second argument must be a list, got {}", args[1])),
        };
        list1.borrow_mut().extend(list2.borrow().clone());
        Ok(args[0].clone())  // Return the list itself
    });
    env.borrow_mut().set("extend".to_string(), extend_fn);

    let insert_fn = Object::NativeFn(|args| {
        if args.len() != 3 {
            return Err("insert() takes exactly 3 arguments (list, index, item)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("insert() first argument must be a list, got {}", args[0])),
        };
        let list_len = list.borrow().len();
        let index = match &args[1] {
            Object::Integer(i) => {
                let mut idx = *i;
                if idx < 0 {
                    idx += list_len as i64;
                }
                idx.clamp(0, list_len as i64) as usize
            },
            _ => return Err(format!("insert() second argument must be an integer, got {}", args[1])),
        };
        let item = args[2].clone();
        list.borrow_mut().insert(index, item);
        Ok(args[0].clone())  // Return the list itself
    });
    env.borrow_mut().set("insert".to_string(), insert_fn);

    let isspace_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("isspace() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| c.is_whitespace()))),
            _ => Err(format!("isspace() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isspace".to_string(), isspace_fn);

    let radians_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("radians() takes exactly 1 argument".to_string());
        }
        let deg = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("radians() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(deg.to_radians()))
    });
    env.borrow_mut().set("radians".to_string(), radians_fn);

    let degrees_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("degrees() takes exactly 1 argument".to_string());
        }
        let rad = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("degrees() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(rad.to_degrees()))
    });
    env.borrow_mut().set("degrees".to_string(), degrees_fn);

    let trunc_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("trunc() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Integer(*f as i64)),
            _ => Err(format!("trunc() argument must be numeric, got {}", args[0])),
        }
    });
    env.borrow_mut().set("trunc".to_string(), trunc_fn);

    let pop_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 3 {
            return Err("pop() takes 1 to 3 arguments".to_string());
        }
        match &args[0] {
            Object::List(list) => {
                if args.len() > 2 {
                    return Err("pop() for list takes 1 or 2 arguments (list, [index])".to_string());
                }
                let list_len = list.borrow().len();
                if list_len == 0 {
                    return Err("pop() from empty list".to_string());
                }
                let index = if args.len() == 2 {
                    match &args[1] {
                        Object::Integer(i) => {
                            let mut idx = *i;
                            if idx < 0 {
                                idx += list_len as i64;
                            }
                            if idx < 0 || idx >= list_len as i64 {
                                return Err(format!("pop() index {} out of range", i));
                            }
                            idx as usize
                        },
                        _ => return Err(format!("pop() second argument must be an integer for list, got {}", args[1])),
                    }
                } else {
                    list_len - 1
                };
                let item = list.borrow_mut().remove(index);
                Ok(item)
            },
            Object::Dictionary(dict) => {
                if args.len() < 2 || args.len() > 3 {
                    return Err("pop() for dictionary takes 2 or 3 arguments (dict, key, [default])".to_string());
                }
                let key = &args[1];
                let mut d = dict.borrow_mut();
                if let Some(val) = d.remove(key) {
                    Ok(val)
                } else if args.len() == 3 {
                    Ok(args[2].clone())
                } else {
                    Err(format!("pop(): key '{}' not found in dictionary", key))
                }
            },
            Object::Set(set) => {
                if args.len() != 1 {
                    return Err("pop() for set takes exactly 1 argument (set)".to_string());
                }
                let mut s = set.borrow_mut();
                if let Some(item) = s.iter().next().cloned() {
                    s.remove(&item);
                    Ok(item)
                } else {
                    Err("pop() from empty set".to_string())
                }
            },
            _ => Err(format!("pop() first argument must be a list, dictionary, or set, got {}", args[0])),
        }
    });
    env.borrow_mut().set("pop".to_string(), pop_fn.clone());
    env.borrow_mut().set("pop_at".to_string(), pop_fn);

    let remove_val_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("remove() takes exactly 2 arguments (collection, item)".to_string());
        }
        match &args[0] {
            Object::List(list) => {
                let target = &args[1];
                let pos = list.borrow().iter().position(|x| x == target);
                if let Some(idx) = pos {
                    list.borrow_mut().remove(idx);
                    Ok(args[0].clone())
                } else {
                    Err(format!("remove(list, x): x not in list"))
                }
            },
            Object::Set(set) => {
                let mut s = set.borrow_mut();
                if s.remove(&args[1]) {
                    Ok(args[0].clone())
                } else {
                    Err(format!("remove(set, x): x not in set"))
                }
            },
            _ => Err(format!("remove() first argument must be a list or set, got {}", args[0])),
        }
    });
    env.borrow_mut().set("remove".to_string(), remove_val_fn);

    let index_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 {
            return Err("index() takes 2 to 4 arguments (item, [start], [end])".to_string());
        }
        match &args[0] {
            Object::List(list) => {
                let list_borrow = list.borrow();
                let len = list_borrow.len();
                let target = &args[1];
                let start = if args.len() >= 3 {
                    match &args[2] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("index() start must be an integer".to_string()),
                    }
                } else { 0 };
                let end = if args.len() == 4 {
                    match &args[3] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("index() end must be an integer".to_string()),
                    }
                } else { len };

                if start < end && start < len {
                    let pos = list_borrow[start..end].iter().position(|x| x == target);
                    match pos {
                        Some(p) => return Ok(Object::Integer((p + start) as i64)),
                        None => {}
                    }
                }
                Err(format!("index(): {} not in list", target))
            },
            Object::String(s) => {
                let len = s.chars().count();
                let sub = match &args[1] {
                    Object::String(val) => val,
                    _ => return Err(format!("index() second argument must be a string, got {}", args[1])),
                };
                let start = if args.len() >= 3 {
                    match &args[2] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("index() start must be an integer".to_string()),
                    }
                } else { 0 };
                let end = if args.len() == 4 {
                    match &args[3] {
                        Object::Integer(i) => {
                            let mut val = *i;
                            if val < 0 { val += len as i64; }
                            val.clamp(0, len as i64) as usize
                        }
                        _ => return Err("index() end must be an integer".to_string()),
                    }
                } else { len };

                if start < end && start < len {
                    let sliced: String = s.chars().skip(start).take(end - start).collect();
                    let idx = sliced.find(sub);
                    match idx {
                        Some(i) => {
                            // Find byte offset i and convert it to char offset within 'sliced'
                            let char_offset = sliced[..i].chars().count();
                            return Ok(Object::Integer((char_offset + start) as i64));
                        }
                        None => {}
                    }
                }
                Err(format!("index(): '{}' not in string", sub))
            },
            _ => Err(format!("index() first argument must be a list or string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("index".to_string(), index_fn);

    let title_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("title() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => {
                let titled: String = s.split_whitespace()
                    .map(|word| {
                        let mut chars = word.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(" ");
                Ok(Object::String(titled))
            },
            _ => Err(format!("title() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("title".to_string(), title_fn);

    let swapcase_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("swapcase() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::String(s) => {
                let swapped: String = s.chars().map(|c| {
                    if c.is_uppercase() { c.to_lowercase().to_string() }
                    else { c.to_uppercase().to_string() }
                }).collect();
                Ok(Object::String(swapped))
            },
            _ => Err(format!("swapcase() arg must be string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("swapcase".to_string(), swapcase_fn);

    let maketrans_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 { return Err("maketrans() takes 2 or 3 arguments".to_string()); }
        let from = match &args[0] { Object::String(s) => s, _ => return Err("maketrans() arg 1 must be string".to_string()) };
        let to = match &args[1] { Object::String(s) => s, _ => return Err("maketrans() arg 2 must be string".to_string()) };
        if from.chars().count() != to.chars().count() { return Err("maketrans() arguments 1 and 2 must be same length".to_string()); }
        let mut map = HashMap::new();
        for (f, t) in from.chars().zip(to.chars()) {
            map.insert(Object::String(f.to_string()), Object::String(t.to_string()));
        }
        if args.len() == 3 {
             let delete = match &args[2] { Object::String(s) => s, _ => return Err("maketrans() arg 3 must be string".to_string()) };
             for c in delete.chars() {
                 map.insert(Object::String(c.to_string()), Object::Null);
             }
        }
        Ok(Object::Dictionary(Rc::new(RefCell::new(map))))
    });
    env.borrow_mut().set("maketrans".to_string(), maketrans_fn);

    let translate_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("translate() takes exactly 2 arguments (string, table)".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("translate() arg 1 must be string".to_string()) };
        let table = match &args[1] { Object::Dictionary(val) => val, _ => return Err("translate() arg 2 must be dictionary".to_string()) };
        let mut res = String::new();
        let t = table.borrow();
        for c in s.chars() {
            let key = Object::String(c.to_string());
            if let Some(val) = t.get(&key) {
                match val {
                    Object::String(v) => res.push_str(v),
                    Object::Null => {},
                    _ => res.push(c),
                }
            } else {
                res.push(c);
            }
        }
        Ok(Object::String(res))
    });
    env.borrow_mut().set("translate".to_string(), translate_fn);



    let ljust_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 { return Err("ljust() takes 2 or 3 arguments (string, width, [fillchar])".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("ljust() arg 1 must be string".to_string()) };
        let width = match &args[1] { Object::Integer(i) => *i as usize, _ => return Err("ljust() arg 2 must be int".to_string()) };
        let fill = if args.len() == 3 {
             match &args[2] { Object::String(val) if val.chars().count() == 1 => val.clone(), _ => return Err("ljust() fillchar must be 1-char string".to_string()) }
        } else { " ".to_string() };
        if s.len() >= width { Ok(Object::String(s.clone())) }
        else { Ok(Object::String(format!("{}{}", s, fill.repeat(width - s.len())))) }
    });
    env.borrow_mut().set("ljust".to_string(), ljust_fn);

    let rjust_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 { return Err("rjust() takes 2 or 3 arguments (string, width, [fillchar])".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("rjust() arg 1 must be string".to_string()) };
        let width = match &args[1] { Object::Integer(i) => *i as usize, _ => return Err("rjust() arg 2 must be int".to_string()) };
        let fill = if args.len() == 3 {
             match &args[2] { Object::String(val) if val.chars().count() == 1 => val.clone(), _ => return Err("rjust() fillchar must be 1-char string".to_string()) }
        } else { " ".to_string() };
        if s.len() >= width { Ok(Object::String(s.clone())) }
        else { Ok(Object::String(format!("{}{}", fill.repeat(width - s.len()), s))) }
    });
    env.borrow_mut().set("rjust".to_string(), rjust_fn);

    let removesuffix_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("removesuffix() takes exactly 2 arguments (string, suffix)".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("removesuffix() arg 1 must be string".to_string()) };
        let suffix = match &args[1] { Object::String(val) => val, _ => return Err("removesuffix() arg 2 must be string".to_string()) };
        if s.ends_with(suffix) { Ok(Object::String(s[..s.len() - suffix.len()].to_string())) }
        else { Ok(Object::String(s.clone())) }
    });
    env.borrow_mut().set("removesuffix".to_string(), removesuffix_fn);

    let removeprefix_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("removeprefix() takes exactly 2 arguments (string, prefix)".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("removeprefix() arg 1 must be string".to_string()) };
        let prefix = match &args[1] { Object::String(val) => val, _ => return Err("removeprefix() arg 2 must be string".to_string()) };
        if s.starts_with(prefix) { Ok(Object::String(s[prefix.len()..].to_string())) }
        else { Ok(Object::String(s.clone())) }
    });
    env.borrow_mut().set("removeprefix".to_string(), removeprefix_fn);

    let partition_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("partition() takes exactly 2 arguments (string, sep)".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("partition() arg 1 must be string".to_string()) };
        let sep = match &args[1] { Object::String(val) => val, _ => return Err("partition() arg 2 must be string".to_string()) };
        if let Some(pos) = s.find(sep) {
            let before = &s[..pos];
            let after = &s[pos + sep.len()..];
            Ok(Object::List(Rc::new(RefCell::new(vec![Object::String(before.to_string()), Object::String(sep.clone()), Object::String(after.to_string())]))))
        } else {
            Ok(Object::List(Rc::new(RefCell::new(vec![Object::String(s.clone()), Object::String("".to_string()), Object::String("".to_string())]))))
        }
    });
    env.borrow_mut().set("partition".to_string(), partition_fn);

    let rfind_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 {
            return Err("rfind() takes 2 to 4 arguments (substring, [start], [end])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("rfind() first argument must be a string, got {}", args[0])),
        };
        let sub = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("rfind() second argument must be a string, got {}", args[1])),
        };
        let len = s.chars().count();
        let start = if args.len() >= 3 {
            match &args[2] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("rfind() start must be an integer".to_string()),
            }
        } else { 0 };
        let end = if args.len() == 4 {
            match &args[3] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("rfind() end must be an integer".to_string()),
            }
        } else { len };

        if start < end && start < len {
            let sliced: String = s.chars().skip(start).take(end - start).collect();
            match sliced.rfind(sub) {
                Some(idx) => {
                    let char_offset = sliced[..idx].chars().count();
                    return Ok(Object::Integer((char_offset + start) as i64));
                }
                None => {}
            }
        }
        Ok(Object::Integer(-1))
    });
    env.borrow_mut().set("rfind".to_string(), rfind_fn);

    let rindex_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 {
            return Err("rindex() takes 2 to 4 arguments (substring, [start], [end])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("rindex() first argument must be a string, got {}", args[0])),
        };
        let sub = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("rindex() second argument must be a string, got {}", args[1])),
        };
        let len = s.chars().count();
        let start = if args.len() >= 3 {
            match &args[2] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("rindex() start must be an integer".to_string()),
            }
        } else { 0 };
        let end = if args.len() == 4 {
            match &args[3] {
                Object::Integer(i) => {
                    let mut val = *i;
                    if val < 0 { val += len as i64; }
                    val.clamp(0, len as i64) as usize
                }
                _ => return Err("rindex() end must be an integer".to_string()),
            }
        } else { len };

        if start < end && start < len {
            let sliced: String = s.chars().skip(start).take(end - start).collect();
            match sliced.rfind(sub) {
                Some(idx) => {
                    let char_offset = sliced[..idx].chars().count();
                    return Ok(Object::Integer((char_offset + start) as i64));
                }
                None => {}
            }
        }
        Err(format!("rindex(): '{}' not in string", sub))
    });
    env.borrow_mut().set("rindex".to_string(), rindex_fn);

    let lstrip_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("lstrip() takes 1 or 2 arguments (string, [chars])".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("lstrip() arg 1 must be string".to_string()) };
        let chars = if args.len() == 2 {
            match &args[1] { Object::String(val) => Some(val), _ => return Err("lstrip() arg 2 must be string".to_string()) }
        } else { None };
        if let Some(c) = chars { Ok(Object::String(s.trim_start_matches(|ch| c.contains(ch)).to_string())) }
        else { Ok(Object::String(s.trim_start().to_string())) }
    });
    env.borrow_mut().set("lstrip".to_string(), lstrip_fn);

    let rstrip_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("rstrip() takes 1 or 2 arguments (string, [chars])".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("rstrip() arg 1 must be string".to_string()) };
        let chars = if args.len() == 2 {
            match &args[1] { Object::String(val) => Some(val), _ => return Err("rstrip() arg 2 must be string".to_string()) }
        } else { None };
        if let Some(c) = chars { Ok(Object::String(s.trim_end_matches(|ch| c.contains(ch)).to_string())) }
        else { Ok(Object::String(s.trim_end().to_string())) }
    });
    env.borrow_mut().set("rstrip".to_string(), rstrip_fn);

    let rpartition_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("rpartition() takes exactly 2 arguments (string, sep)".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("rpartition() arg 1 must be string".to_string()) };
        let sep = match &args[1] { Object::String(val) => val, _ => return Err("rpartition() arg 2 must be string".to_string()) };
        if let Some(pos) = s.rfind(sep) {
            let before = &s[..pos];
            let after = &s[pos + sep.len()..];
            Ok(Object::List(Rc::new(RefCell::new(vec![Object::String(before.to_string()), Object::String(sep.clone()), Object::String(after.to_string())]))))
        } else {
            Ok(Object::List(Rc::new(RefCell::new(vec![Object::String("".to_string()), Object::String("".to_string()), Object::String(s.clone())]))))
        }
    });
    env.borrow_mut().set("rpartition".to_string(), rpartition_fn);

    let splitlines_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("splitlines() takes 1 or 2 arguments (string, [keepends])".to_string()); }
        let s = match &args[0] { Object::String(val) => val, _ => return Err("splitlines() arg 1 must be string".to_string()) };
        let keepends = if args.len() == 2 { args[1].is_truthy() } else { false };
        let mut lines = Vec::new();
        if keepends {
            let mut start = 0;
            for (i, c) in s.char_indices() {
                if c == '\n' {
                    lines.push(Object::String(s[start..=i].to_string()));
                    start = i + 1;
                } else if c == '\r' && s.as_bytes().get(i+1) == Some(&b'\n') {
                    // Handled by \n next iteration
                } else if c == '\r' {
                     lines.push(Object::String(s[start..=i].to_string()));
                     start = i + 1;
                }
            }
            if start < s.len() { lines.push(Object::String(s[start..].to_string())); }
        } else {
            for line in s.lines() { lines.push(Object::String(line.to_string())); }
        }
        Ok(Object::List(Rc::new(RefCell::new(lines))))
    });
    env.borrow_mut().set("splitlines".to_string(), splitlines_fn);

    let factorial_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("factorial() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("factorial() argument must be an integer, got {}", args[0])),
        };
        if n < 0 {
            return Err("factorial() argument must be non-negative".to_string());
        }
        let mut res: i64 = 1;
        for i in 1..=n {
            res = res.checked_mul(i).ok_or("factorial() result overflow")?;
        }
        Ok(Object::Integer(res))
    });
    env.borrow_mut().set("factorial".to_string(), factorial_fn);

    let gcd_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("gcd() takes exactly 2 arguments".to_string());
        }
        let mut a = match &args[0] {
            Object::Integer(i) => i.abs(),
            _ => return Err(format!("gcd() arguments must be integers, got {}", args[0])),
        };
        let mut b = match &args[1] {
            Object::Integer(i) => i.abs(),
            _ => return Err(format!("gcd() arguments must be integers, got {}", args[1])),
        };
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        Ok(Object::Integer(a))
    });
    env.borrow_mut().set("gcd".to_string(), gcd_fn);

    let lcm_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("lcm() takes exactly 2 arguments".to_string()); }
        let a = match &args[0] { Object::Integer(i) => i.abs(), _ => return Err("lcm() arg 1 must be int".to_string()) };
        let b = match &args[1] { Object::Integer(i) => i.abs(), _ => return Err("lcm() arg 2 must be int".to_string()) };
        if a == 0 || b == 0 { return Ok(Object::Integer(0)); }
        // (a*b)/gcd(a,b)
        let mut x = a;
        let mut y = b;
        while y != 0 { let t = y; y = x % y; x = t; }
        let gcd = x;
        Ok(Object::Integer((a / gcd) * b))
    });
    env.borrow_mut().set("lcm".to_string(), lcm_fn);

    let copysign_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("copysign() takes exactly 2 arguments".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("copysign() first arg must be numeric".to_string()) };
        let y = match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("copysign() second arg must be numeric".to_string()) };
        Ok(Object::Float(x.abs() * y.signum()))
    });
    env.borrow_mut().set("copysign".to_string(), copysign_fn);

    let remainder_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("remainder() takes exactly 2 arguments".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("remainder() first arg must be numeric".to_string()) };
        let y = match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("remainder() second arg must be numeric".to_string()) };
        if y == 0.0 { return Err("remainder() divisor cannot be zero".to_string()); }
        let res = x - (x / y).round() * y;
        Ok(Object::Float(res))
    });
    env.borrow_mut().set("remainder".to_string(), remainder_fn);

    let ldexp_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("ldexp() takes exactly 2 arguments".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("ldexp() first arg must be numeric".to_string()) };
        let i = match args[1] { Object::Integer(i) => i as i32, _ => return Err("ldexp() second arg must be an integer".to_string()) };
        Ok(Object::Float(x * 2.0f64.powi(i)))
    });
    env.borrow_mut().set("ldexp".to_string(), ldexp_fn);

    let frexp_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("frexp() takes exactly 1 argument".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("frexp() argument must be numeric".to_string()) };
        if x == 0.0 {
            return Ok(Object::List(Rc::new(RefCell::new(vec![Object::Float(0.0), Object::Integer(0)]))));
        }
        let bits = x.to_bits();
        // Extract exponent (bits 52-62)
        let exp = ((bits >> 52) & 0x7ff) as i32 - 1022;
        // Construct mantissa with exponent biased to 1022 (0.5 to 1.0)
        let mantissa_bits = (bits & 0x800fffffffffffff) | (1022 << 52);
        let mantissa = f64::from_bits(mantissa_bits);
        
        Ok(Object::List(Rc::new(RefCell::new(vec![Object::Float(mantissa), Object::Integer(exp as i64)]))))
    });
    env.borrow_mut().set("frexp".to_string(), frexp_fn);

    let modf_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("modf() takes exactly 1 argument".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("modf() argument must be numeric".to_string()) };
        let i = x.trunc();
        let f = x - i;
        Ok(Object::List(Rc::new(RefCell::new(vec![Object::Float(f), Object::Float(i)]))))
    });
    env.borrow_mut().set("modf".to_string(), modf_fn);

    let isclose_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 4 { return Err("isclose() takes 2 to 4 arguments (a, b, [rel_tol], [abs_tol])".to_string()); }
        let a = match &args[0] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("isclose() a must be numeric".to_string()) };
        let b = match &args[1] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("isclose() b must be numeric".to_string()) };
        let rel_tol = if args.len() >= 3 { match &args[2] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => 1e-09 } } else { 1e-09 };
        let abs_tol = if args.len() == 4 { match &args[3] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => 0.0 } } else { 0.0 };
        
        let diff = (a - b).abs();
        Ok(Object::Boolean(diff <= (rel_tol * a.abs()).max(rel_tol * b.abs()).max(abs_tol)))
    });
    env.borrow_mut().set("isclose".to_string(), isclose_fn);

    let bool_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("bool() takes exactly 1 argument".to_string()); }
        Ok(Object::Boolean(args[0].is_truthy()))
    });
    env.borrow_mut().set("bool".to_string(), bool_fn);

    fn flatten_recursive(obj: &Object) -> Vec<Object> {
        match obj {
            Object::List(l) => {
                let mut res = Vec::new();
                for item in l.borrow().iter() {
                    res.extend(flatten_recursive(item));
                }
                res
            },
            _ => vec![obj.clone()],
        }
    }

    let flatten_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("flatten() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Tensor(t.flatten())),
            Object::List(_) => Ok(Object::List(Rc::new(RefCell::new(flatten_recursive(&args[0]))))),
            _ => Ok(args[0].clone()),
        }
    });
    env.borrow_mut().set("flatten".to_string(), flatten_fn);

    let copy_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("copy() takes exactly 1 argument".to_string()); }
        Ok(args[0].deep_copy())
    });
    env.borrow_mut().set("copy".to_string(), copy_fn);

    let list_remove_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("remove() takes exactly 2 arguments (list, value)".to_string()); }
        let list = match &args[0] {
            Object::List(l) => l,
            _ => return Err("remove() first argument must be a list".to_string()),
        };
        let mut l = list.borrow_mut();
        if let Some(pos) = l.iter().position(|x| x == &args[1]) {
            l.remove(pos);
            Ok(Object::Null)
        } else {
            Err(format!("remove(x): x not in list"))
        }
    });
    env.borrow_mut().set("remove".to_string(), list_remove_fn);

    let clear_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("clear() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::List(l) => { l.borrow_mut().clear(); Ok(Object::Null) },
            Object::Dictionary(d) => { d.borrow_mut().clear(); Ok(Object::Null) },
            Object::Set(s) => { s.borrow_mut().clear(); Ok(Object::Null) },
            _ => Err("clear() argument must be list, dict, or set".to_string()),
        }
    });
    env.borrow_mut().set("clear".to_string(), clear_fn);

    let dict_update_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("update() takes exactly 2 arguments (dict, other)".to_string()); }
        let dict = match &args[0] {
            Object::Dictionary(d) => d,
            _ => return Err("update() first argument must be a dictionary".to_string()),
        };
        let other = match &args[1] {
            Object::Dictionary(d) => d,
            _ => return Err("update() second argument must be a dictionary".to_string()),
        };
        for (k, v) in other.borrow().iter() {
            dict.borrow_mut().insert(k.clone(), v.clone());
        }
        Ok(Object::Null)
    });
    env.borrow_mut().set("update".to_string(), dict_update_fn);

    let id_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("id() takes exactly 1 argument".to_string()); }
        let addr = match &args[0] {
            Object::List(l) => Rc::as_ptr(l) as usize,
            Object::Dictionary(d) => Rc::as_ptr(d) as usize,
            Object::Set(s) => Rc::as_ptr(s) as usize,
            Object::Function { env, .. } => Rc::as_ptr(env) as usize,
            _ => &args[0] as *const _ as usize,
        };
        Ok(Object::Integer(addr as i64))
    });
    env.borrow_mut().set("id".to_string(), id_fn);

    let chr_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("chr() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("chr() argument must be an integer, got {}", args[0])),
        };
        match std::char::from_u32(n as u32) {
            Some(c) => Ok(Object::String(c.to_string())),
            None => Err(format!("chr() argument {} is not a valid Unicode code point", n)),
        }
    });
    env.borrow_mut().set("chr".to_string(), chr_fn);

    let ord_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("ord() takes exactly 1 argument".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("ord() argument must be a string, got {}", args[0])),
        };
        if s.chars().count() != 1 {
            return Err(format!("ord() expected a character, but string of length {} found", s.len()));
        }
        let c = s.chars().next().unwrap();
        Ok(Object::Integer(c as i64))
    });
    env.borrow_mut().set("ord".to_string(), ord_fn);

    let bin_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("bin() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::String(format!("0b{:b}", i))),
            _ => Err(format!("bin() argument must be an integer, got {}", args[0])),
        }
    });
    env.borrow_mut().set("bin".to_string(), bin_fn);

    let oct_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("oct() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::String(format!("0o{:o}", i))),
            _ => Err(format!("oct() argument must be an integer, got {}", args[0])),
        }
    });
    env.borrow_mut().set("oct".to_string(), oct_fn);

    let hex_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("hex() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Integer(i) => Ok(Object::String(format!("0x{:x}", i))),
            _ => Err(format!("hex() argument must be an integer, got {}", args[0])),
        }
    });
    env.borrow_mut().set("hex".to_string(), hex_fn);

    let choice_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("choice() takes exactly 1 argument (list)".to_string()); }
        match &args[0] {
            Object::List(l) => {
                let items = l.borrow();
                if items.is_empty() { return Err("choice() cannot select from empty list".to_string()); }
                let idx = rand::thread_rng().gen_range(0..items.len());
                Ok(items[idx].clone())
            },
            _ => Err(format!("choice() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("choice".to_string(), choice_fn);

    let shuffle_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("shuffle() takes exactly 1 argument (list)".to_string()); }
        match &args[0] {
            Object::List(l) => {
                use rand::seq::SliceRandom;
                l.borrow_mut().shuffle(&mut rand::thread_rng());
                Ok(Object::Null)
            },
            _ => Err(format!("shuffle() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("shuffle".to_string(), shuffle_fn);

    let casefold_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("casefold() takes exactly 1 argument (string)".to_string()); }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.to_lowercase())),
            _ => Err(format!("casefold() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("casefold".to_string(), casefold_fn);

    let ascii_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("ascii() takes exactly 1 argument".to_string()); }
        let escaped = match &args[0] {
            Object::String(s) => s.escape_debug().to_string(),
            _ => args[0].to_string().escape_debug().to_string(),
        };
        Ok(Object::String(format!("'{}'", escaped)))
    });
    env.borrow_mut().set("ascii".to_string(), ascii_fn);

    let clip_fn = Object::NativeFn(|args| {
        if args.len() != 3 { return Err("clip() takes 3 arguments (tensor, min, max)".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("clip() arg 1 must be tensor".to_string()) };
        let min = match &args[1] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("clip() min must be numeric".to_string()) };
        let max = match &args[2] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("clip() max must be numeric".to_string()) };
        Ok(Object::Tensor(t.clip(min, max)))
    });
    env.borrow_mut().set("clip".to_string(), clip_fn);

    let norm_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("norm() takes 1 or 2 arguments (tensor, [p])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("norm() arg 1 must be tensor".to_string()) };
        let p = if args.len() == 2 {
            match &args[1] { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("norm() p must be numeric".to_string()) }
        } else { 2.0 };
        Ok(Object::Float(t.norm(p)))
    });
    env.borrow_mut().set("norm".to_string(), norm_fn);

    let dir_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("dir() takes exactly 1 argument".to_string()); }
        let methods = match &args[0] {
            Object::List(_) => vec!["append", "extend", "pop", "remove", "insert", "clear", "reverse", "sort", "index", "count", "swap", "unique", "flat"],
            Object::String(_) => vec!["upper", "lower", "capitalize", "title", "swapcase", "casefold", "strip", "lstrip", "rstrip", "startswith", "endswith", "replace", "split", "splitlines", "join", "find", "rfind", "index", "rindex", "count"],
            Object::Tensor(_) => vec!["shape", "ndim", "len", "reshape", "transpose", "squeeze", "unsqueeze", "argmax", "argmin", "sum", "mean", "std", "var", "clip", "norm", "diag", "trace", "item", "fill", "sqrt", "exp", "log"],
            Object::Dictionary(_) => vec!["keys", "values", "items", "get", "update", "pop", "popitem", "clear", "setdefault", "fromkeys"],
            Object::Set(_) => vec!["add", "discard", "clear", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint"],
            _ => vec![],
        };
        let mut res: Vec<Object> = methods.into_iter().map(|s| Object::String(s.to_string())).collect();
        res.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
        Ok(Object::List(Rc::new(RefCell::new(res))))
    });
    env.borrow_mut().set("dir".to_string(), dir_fn);

    let nan_to_num_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 4 { return Err("nan_to_num() takes 1 to 4 arguments (x, [nan], [posinf], [neginf])".to_string()); }
        let nan_val = if args.len() >= 2 { match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => 0.0 } } else { 0.0 };
        let posinf_val = if args.len() >= 3 { match args[2] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => f64::MAX } } else { f64::MAX };
        let neginf_val = if args.len() >= 4 { match args[3] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => f64::MIN } } else { f64::MIN };

        let transform = |x: f64| {
            if x.is_nan() { nan_val }
            else if x.is_infinite() && x.is_sign_positive() { posinf_val }
            else if x.is_infinite() && x.is_sign_negative() { neginf_val }
            else { x }
        };

        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Float(transform(*f))),
            Object::Tensor(t) => Ok(Object::Tensor(Tensor { inner: t.inner.mapv(transform) })),
            _ => Err("nan_to_num() argument must be numeric or tensor".to_string()),
        }
    });
    env.borrow_mut().set("nan_to_num".to_string(), nan_to_num_fn);

    let cbrt_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("cbrt() takes exactly 1 argument".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("cbrt() arg must be numeric".to_string()) };
        Ok(Object::Float(x.cbrt()))
    });
    env.borrow_mut().set("cbrt".to_string(), cbrt_fn);

    let log1p_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("log1p() takes exactly 1 argument".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("log1p() arg must be numeric".to_string()) };
        Ok(Object::Float(x.ln_1p()))
    });
    env.borrow_mut().set("log1p".to_string(), log1p_fn);

    let expm1_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("expm1() takes exactly 1 argument".to_string()); }
        let x = match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("expm1() arg must be numeric".to_string()) };
        Ok(Object::Float(x.exp_m1()))
    });
    env.borrow_mut().set("expm1".to_string(), expm1_fn);

    let item_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("item() takes exactly 1 argument (tensor)".to_string()); }
        match &args[0] {
            Object::Tensor(t) => t.item().map(Object::Float),
            _ => Err("item() argument must be a tensor".to_string()),
        }
    });
    env.borrow_mut().set("item".to_string(), item_fn);

    let fill_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("fill() takes exactly 2 arguments (tensor, value)".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("fill() first arg must be a tensor".to_string()) };
        let val = match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("fill() second arg must be numeric".to_string()) };
        Ok(Object::Tensor(t.fill(val)))
    });
    env.borrow_mut().set("fill".to_string(), fill_fn);

    let arange_fn = Object::NativeFn(|args| {
        let (start, stop, step) = match args.len() {
            1 => (0.0, match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg must be numeric".to_string()) }, 1.0),
            2 => (
                match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg 1 must be numeric".to_string()) },
                match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg 2 must be numeric".to_string()) },
                1.0
            ),
            3 => (
                match args[0] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg 1 must be numeric".to_string()) },
                match args[1] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg 2 must be numeric".to_string()) },
                match args[2] { Object::Integer(i) => i as f64, Object::Float(f) => f, _ => return Err("arange() arg 3 must be numeric".to_string()) }
            ),
            _ => return Err("arange() takes 1 to 3 arguments".to_string()),
        };
        Ok(Object::Tensor(Tensor::arange(start, stop, step)))
    });
    env.borrow_mut().set("arange".to_string(), arange_fn);

    let cumsum_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("cumsum() takes 1 or 2 arguments (tensor, [axis])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("cumsum() arg 1 must be tensor".to_string()) };
        let axis = if args.len() == 2 {
            match args[1] { Object::Integer(i) => Some(i as usize), _ => return Err("cumsum() axis must be integer".to_string()) }
        } else { None };
        Ok(Object::Tensor(t.cumsum(axis)))
    });
    env.borrow_mut().set("cumsum".to_string(), cumsum_fn);

    let cumprod_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("cumprod() takes 1 or 2 arguments (tensor, [axis])".to_string()); }
        let t = match &args[0] { Object::Tensor(t) => t, _ => return Err("cumprod() arg 1 must be tensor".to_string()) };
        let axis = if args.len() == 2 {
            match args[1] { Object::Integer(i) => Some(i as usize), _ => return Err("cumprod() axis must be integer".to_string()) }
        } else { None };
        Ok(Object::Tensor(t.cumprod(axis)))
    });
    env.borrow_mut().set("cumprod".to_string(), cumprod_fn);

    let bit_length_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("bit_length() takes exactly 1 argument".to_string()); }
        match args[0] {
            Object::Integer(i) => {
                let val = i.abs();
                if val == 0 { return Ok(Object::Integer(0)); }
                Ok(Object::Integer((64 - val.leading_zeros()) as i64))
            },
            _ => Err("bit_length() argument must be an integer".to_string()),
        }
    });
    env.borrow_mut().set("bit_length".to_string(), bit_length_fn);

    let is_integer_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("is_integer() takes exactly 1 argument".to_string()); }
        match args[0] {
            Object::Float(f) => Ok(Object::Boolean(f.fract() == 0.0)),
            Object::Integer(_) => Ok(Object::Boolean(true)),
            _ => Ok(Object::Boolean(false)),
        }
    });
    env.borrow_mut().set("is_integer".to_string(), is_integer_fn);

    let getattr_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 { return Err("getattr() takes 2 or 3 arguments (obj, name, [default])".to_string()); }
        let name = match &args[1] { Object::String(s) => s, _ => return Err("getattr() name must be string".to_string()) };
        let default = if args.len() == 3 { Some(args[2].clone()) } else { None };
        
        let methods = match &args[0] {
            Object::List(_) => vec!["append", "extend", "pop", "remove", "insert", "clear", "reverse", "sort", "index", "count", "swap", "unique", "flat"],
            Object::String(_) => vec!["upper", "lower", "capitalize", "title", "swapcase", "casefold", "strip", "lstrip", "rstrip", "startswith", "endswith", "replace", "split", "splitlines", "join", "find", "rfind", "index", "rindex", "count"],
            Object::Tensor(_) => vec!["shape", "ndim", "len", "reshape", "transpose", "squeeze", "unsqueeze", "argmax", "argmin", "sum", "mean", "std", "var", "clip", "norm", "diag", "trace", "item", "fill", "sqrt", "exp", "log"],
            Object::Dictionary(_) => vec!["keys", "values", "items", "get", "update", "pop", "popitem", "clear", "setdefault", "fromkeys"],
            Object::Set(_) => vec!["add", "discard", "clear", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint"],
            Object::Module { env: module_env, .. } => {
                if let Some(v) = module_env.borrow().get(name) { return Ok(v); }
                vec![] // fallback to default
            }
            _ => vec![],
        };

        if methods.contains(&name.as_str()) {
            // In Flux, methods are just built-in functions. 
            // We can return the native function directly if we can find it in the environment.
            // But this requires a reference to the global environment which we don't have easily in NativeFn closure.
            // Simplified: return the name as a string or a dummy "method" object if we had one.
            // For now, let's just use the current approach of registry.
            // Actually, we can just return the string if it's found, or we can look it up in a static way if we had a static registry.
            // Let's just return a placeholder or error for now as Flux doesn't yet have first-class method objects that are easy to extract.
            return Ok(Object::String(format!("<method '{}'>", name)));
        }

        if let Some(d) = default { Ok(d) }
        else { Err(format!("Attribute '{}' not found on {}", name, args[0])) }
    });
    env.borrow_mut().set("getattr".to_string(), getattr_fn);

    let hasattr_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("hasattr() takes exactly 2 arguments (obj, name)".to_string()); }
        let name = match &args[1] { Object::String(s) => s, _ => return Err("hasattr() name must be string".to_string()) };
        
        let methods = match &args[0] {
            Object::List(_) => vec!["append", "extend", "pop", "remove", "insert", "clear", "reverse", "sort", "index", "count", "swap", "unique", "flat"],
            Object::String(_) => vec!["upper", "lower", "capitalize", "title", "swapcase", "casefold", "strip", "lstrip", "rstrip", "startswith", "endswith", "replace", "split", "splitlines", "join", "find", "rfind", "index", "rindex", "count"],
            Object::Tensor(_) => vec!["shape", "ndim", "len", "reshape", "transpose", "squeeze", "unsqueeze", "argmax", "argmin", "sum", "mean", "std", "var", "clip", "norm", "diag", "trace", "item", "fill", "sqrt", "exp", "log"],
            Object::Dictionary(_) => vec!["keys", "values", "items", "get", "update", "pop", "popitem", "clear", "setdefault", "fromkeys"],
            Object::Set(_) => vec!["add", "discard", "clear", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint"],
            Object::Module { env: module_env, .. } => {
                return Ok(Object::Boolean(module_env.borrow().get(name).is_some()));
            }
            _ => vec![],
        };

        Ok(Object::Boolean(methods.contains(&name.as_str())))
    });
    env.borrow_mut().set("hasattr".to_string(), hasattr_fn);

    let clamp_fn = Object::NativeFn(|args| {
        if args.len() != 3 {
            return Err("clamp() takes exactly 3 arguments (val, min, max)".to_string());
        }
        let val = &args[0];
        let min = &args[1];
        let max = &args[2];
        
        match (val, min, max) {
            (Object::Integer(v), Object::Integer(mi), Object::Integer(ma)) => {
                Ok(Object::Integer((*v).clamp(*mi, *ma)))
            },
            (v, mi, ma) => {
                let vf = match v { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("clamp() args must be numeric".to_string()) };
                let mif = match mi { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("clamp() args must be numeric".to_string()) };
                let maf = match ma { Object::Integer(i) => *i as f64, Object::Float(f) => *f, _ => return Err("clamp() args must be numeric".to_string()) };
                Ok(Object::Float(vf.clamp(mif, maf)))
            }
        }
    });
    env.borrow_mut().set("clamp".to_string(), clamp_fn);

    let swap_fn = Object::NativeFn(|args| {
        if args.len() != 3 {
            return Err("swap() takes exactly 3 arguments (list, i, j)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("swap() first argument must be a list, got {}", args[0])),
        };
        let i = match &args[1] {
            Object::Integer(idx) => *idx as usize,
            _ => return Err(format!("swap() second argument must be an integer, got {}", args[1])),
        };
        let j = match &args[2] {
            Object::Integer(idx) => *idx as usize,
            _ => return Err(format!("swap() third argument must be an integer, got {}", args[2])),
        };
        let list_borrow = list.borrow();
        if i >= list_borrow.len() || j >= list_borrow.len() {
            return Err(format!("swap() index out of range: list length {}, indices {}, {}", list_borrow.len(), i, j));
        }
        let mut new_list = list_borrow.clone();
        new_list.swap(i, j);
        Ok(Object::List(Rc::new(RefCell::new(new_list))))
    });
    env.borrow_mut().set("swap".to_string(), swap_fn);

    let zfill_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("zfill() takes exactly 2 arguments (string, width)".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("zfill() first argument must be a string, got {}", args[0])),
        };
        let width = match &args[1] {
            Object::Integer(i) => *i as usize,
            _ => return Err(format!("zfill() second argument must be an integer, got {}", args[1])),
        };
        if s.len() >= width {
            return Ok(Object::String(s.clone()));
        }
        let padding_len = width - s.len();
        let padding = "0".repeat(padding_len);
        Ok(Object::String(format!("{}{}", padding, s)))
    });
    env.borrow_mut().set("zfill".to_string(), zfill_fn);

    let center_fn = Object::NativeFn(|args| {
        if args.len() < 2 || args.len() > 3 {
            return Err("center() takes 2 or 3 arguments (string, width, [fillchar])".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("center() first argument must be a string, got {}", args[0])),
        };
        let width = match &args[1] {
            Object::Integer(i) => *i as usize,
            _ => return Err(format!("center() second argument must be an integer, got {}", args[1])),
        };
        let fill = if args.len() == 3 {
            match &args[2] {
                Object::String(val) => {
                    if val.chars().count() != 1 {
                        return Err("center() fillchar must be a single character".to_string());
                    }
                    val.clone()
                },
                _ => return Err("center() fillchar must be a string".to_string()),
            }
        } else {
            " ".to_string()
        };

        if s.len() >= width {
            return Ok(Object::String(s.clone()));
        }

        let total_pad = width - s.len();
        let left_pad = total_pad / 2;
        let right_pad = total_pad - left_pad;
        
        Ok(Object::String(format!(
            "{}{}{}",
            fill.repeat(left_pad),
            s,
            fill.repeat(right_pad)
        )))
    });
    env.borrow_mut().set("center".to_string(), center_fn);

    let time_fn = Object::NativeFn(|_| {
        let start = SystemTime::now();
        let since_the_epoch = start.duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Ok(Object::Float(since_the_epoch.as_secs_f64()))
    });
    env.borrow_mut().set("time".to_string(), time_fn);

    let isqrt_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("isqrt() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("isqrt() argument must be an integer, got {}", args[0])),
        };
        if n < 0 {
            return Err("isqrt() argument must be non-negative".to_string());
        }
        Ok(Object::Integer((n as f64).sqrt() as i64))
    });
    env.borrow_mut().set("isqrt".to_string(), isqrt_fn);

    let hypot_fn = Object::NativeFn(|args| {
        if args.len() == 0 {
            return Ok(Object::Float(0.0));
        }
        let mut sum_sq: f64 = 0.0;
        for arg in args.iter() {
            let val = match arg {
                Object::Integer(i) => *i as f64,
                Object::Float(f) => *f,
                _ => return Err(format!("hypot() arguments must be numeric, got {}", arg)),
            };
            sum_sq += val * val;
        }
        Ok(Object::Float(sum_sq.sqrt()))
    });
    env.borrow_mut().set("hypot".to_string(), hypot_fn);

    let dist_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("dist() takes exactly 2 arguments (p, q)".to_string());
        }
        let p_items = match &args[0] {
            Object::List(l) => l.borrow().clone(),
            _ => return Err(format!("dist() first argument must be a list, got {}", args[0])),
        };
        let q_items = match &args[1] {
            Object::List(l) => l.borrow().clone(),
            _ => return Err(format!("dist() second argument must be a list, got {}", args[1])),
        };
        if p_items.len() != q_items.len() {
            return Err("dist() arguments must be lists of the same length".to_string());
        }
        let mut d_sum_sq: f64 = 0.0;
        for (p_obj, q_obj) in p_items.iter().zip(q_items.iter()) {
            let p_val = match p_obj {
                Object::Integer(i) => *i as f64,
                Object::Float(f) => *f,
                _ => return Err(format!("dist() point coordinates must be numeric, got {}", p_obj)),
            };
            let q_val = match q_obj {
                Object::Integer(i) => *i as f64,
                Object::Float(f) => *f,
                _ => return Err(format!("dist() point coordinates must be numeric, got {}", q_obj)),
            };
            let diff = p_val - q_val;
            d_sum_sq += diff * diff;
        }
        Ok(Object::Float(d_sum_sq.sqrt()))
    });
    env.borrow_mut().set("dist".to_string(), dist_fn);

    let flat_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("flat() takes exactly 1 argument (list)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("flat() argument must be a list, got {}", args[0])),
        };
        let mut flattened = Vec::new();
        for item in list.borrow().iter() {
            match item {
                Object::List(sub_list) => {
                    flattened.extend(sub_list.borrow().clone());
                },
                _ => flattened.push(item.clone()),
            }
        }
        Ok(Object::List(Rc::new(RefCell::new(flattened))))
    });
    env.borrow_mut().set("flat".to_string(), flat_fn);



    let set_add_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("add() takes exactly 2 arguments (set, item)".to_string()); }
        let set = match &args[0] {
            Object::Set(s) => s,
            _ => return Err(format!("add() first argument must be a set, got {}", args[0])),
        };
        set.borrow_mut().insert(args[1].clone());
        Ok(args[0].clone())
    });
    env.borrow_mut().set("add".to_string(), set_add_fn);

    let set_discard_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("discard() takes exactly 2 arguments (set, item)".to_string()); }
        let set = match &args[0] {
            Object::Set(s) => s,
            _ => return Err(format!("discard() first argument must be a set, got {}", args[0])),
        };
        set.borrow_mut().remove(&args[1]);
        Ok(args[0].clone())
    });
    env.borrow_mut().set("discard".to_string(), set_discard_fn);

    let set_union_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("union() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("union() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("union() second arg must be set".to_string()) };
        let mut result = s1.borrow().clone();
        for item in s2.borrow().iter() {
            result.insert(item.clone());
        }
        Ok(Object::Set(Rc::new(RefCell::new(result))))
    });
    env.borrow_mut().set("union".to_string(), set_union_fn);

    let set_intersection_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("intersection() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("intersection() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("intersection() second arg must be set".to_string()) };
        let mut result = std::collections::HashSet::new();
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b1.iter() {
            if b2.contains(item) {
                result.insert(item.clone());
            }
        }
        Ok(Object::Set(Rc::new(RefCell::new(result))))
    });
    env.borrow_mut().set("intersection".to_string(), set_intersection_fn);

    let set_difference_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("difference() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("difference() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("difference() second arg must be set".to_string()) };
        let mut result = std::collections::HashSet::new();
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b1.iter() {
            if !b2.contains(item) {
                result.insert(item.clone());
            }
        }
        Ok(Object::Set(Rc::new(RefCell::new(result))))
    });
    env.borrow_mut().set("difference".to_string(), set_difference_fn);

    let set_symmetric_difference_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("symmetric_difference() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("symmetric_difference() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("symmetric_difference() second arg must be set".to_string()) };
        let mut result = std::collections::HashSet::new();
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b1.iter() {
            if !b2.contains(item) {
                result.insert(item.clone());
            }
        }
        for item in b2.iter() {
            if !b1.contains(item) {
                result.insert(item.clone());
            }
        }
        Ok(Object::Set(Rc::new(RefCell::new(result))))
    });
    env.borrow_mut().set("symmetric_difference".to_string(), set_symmetric_difference_fn);

    let set_issubset_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("issubset() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("issubset() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("issubset() second arg must be set".to_string()) };
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b1.iter() {
            if !b2.contains(item) {
                return Ok(Object::Boolean(false));
            }
        }
        Ok(Object::Boolean(true))
    });
    env.borrow_mut().set("issubset".to_string(), set_issubset_fn);

    let set_issuperset_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("issuperset() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("issuperset() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("issuperset() second arg must be set".to_string()) };
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b2.iter() {
            if !b1.contains(item) {
                return Ok(Object::Boolean(false));
            }
        }
        Ok(Object::Boolean(true))
    });
    env.borrow_mut().set("issuperset".to_string(), set_issuperset_fn);

    let set_isdisjoint_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("isdisjoint() takes exactly 2 arguments (set, other)".to_string()); }
        let s1 = match &args[0] { Object::Set(s) => s, _ => return Err("isdisjoint() first arg must be set".to_string()) };
        let s2 = match &args[1] { Object::Set(s) => s, _ => return Err("isdisjoint() second arg must be set".to_string()) };
        let b1 = s1.borrow();
        let b2 = s2.borrow();
        for item in b1.iter() {
            if b2.contains(item) {
                return Ok(Object::Boolean(false));
            }
        }
        Ok(Object::Boolean(true))
    });
    env.borrow_mut().set("isdisjoint".to_string(), set_isdisjoint_fn);

    let update_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("update() takes exactly 2 arguments (collection, other)".to_string()); }
        match &args[0] {
            Object::Dictionary(dict) => {
                let other = match &args[1] {
                    Object::Dictionary(d) => d,
                    _ => return Err(format!("update() for dictionary second argument must be a dictionary, got {}", args[1])),
                };
                for (k, v) in other.borrow().iter() {
                    dict.borrow_mut().insert(k.clone(), v.clone());
                }
                Ok(args[0].clone())
            },
            Object::Set(set) => {
                let other_items: Vec<Object> = match &args[1] {
                    Object::Set(s) => s.borrow().iter().cloned().collect(),
                    Object::List(l) => l.borrow().clone(),
                    Object::String(s) => s.chars().map(|c| Object::String(c.to_string())).collect(),
                    Object::Dictionary(d) => d.borrow().keys().cloned().collect(),
                    _ => return Err(format!("update() for set second arg must be iterable, got {}", args[1])),
                };

                let mut s_borrow = set.borrow_mut();
                for item in other_items {
                    s_borrow.insert(item);
                }
                Ok(args[0].clone())
            },
            _ => Err(format!("update() first argument must be a dictionary or set, got {}", args[0])),
        }
    });
    env.borrow_mut().set("update".to_string(), update_fn);

    let fromkeys_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("fromkeys() takes 1 or 2 arguments (iterable, [value])".to_string()); }
        let value = if args.len() == 2 { args[1].clone() } else { Object::Null };
        let mut d = std::collections::HashMap::new();
        let items: Vec<Object> = match &args[0] {
            Object::List(l) => l.borrow().clone(),
            Object::Set(s) => s.borrow().iter().cloned().collect(),
            Object::String(s) => s.chars().map(|c| Object::String(c.to_string())).collect(),
            Object::Dictionary(dict) => dict.borrow().keys().cloned().collect(),
            _ => return Err(format!("fromkeys() first argument must be iterable, got {}", args[0])),
        };
        for item in items {
            d.insert(item, value.clone());
        }
        Ok(Object::Dictionary(Rc::new(RefCell::new(d))))
    });
    env.borrow_mut().set("fromkeys".to_string(), fromkeys_fn);

    let setdefault_fn = Object::NativeFn(|args| {
        if args.len() != 3 { return Err("setdefault() takes exactly 3 arguments (dict, key, default)".to_string()); }
        let dict = match &args[0] { Object::Dictionary(val) => val, _ => return Err("setdefault() arg 1 must be dictionary".to_string()) };
        let key = args[1].clone();
        let default = args[2].clone();
        let mut d = dict.borrow_mut();
        let val = d.entry(key).or_insert(default);
        Ok(val.clone())
    });
    env.borrow_mut().set("setdefault".to_string(), setdefault_fn);

    let comb_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("comb() takes exactly 2 arguments (n, k)".to_string()); }
        let n = match &args[0] { Object::Integer(i) => *i, _ => return Err("comb() n must be int".to_string()) };
        let k = match &args[1] { Object::Integer(i) => *i, _ => return Err("comb() k must be int".to_string()) };
        if k < 0 || k > n { return Ok(Object::Integer(0)); }
        let mut res = 1i64;
        let k = k.min(n - k);
        for i in 1..=k {
            res = res * (n - i + 1) / i;
        }
        Ok(Object::Integer(res))
    });
    env.borrow_mut().set("comb".to_string(), comb_fn);

    let perm_fn = Object::NativeFn(|args| {
        if args.len() != 2 { return Err("perm() takes exactly 2 arguments (n, k)".to_string()); }
        let n = match &args[0] { Object::Integer(i) => *i, _ => return Err("perm() n must be int".to_string()) };
        let k = match &args[1] { Object::Integer(i) => *i, _ => return Err("perm() k must be int".to_string()) };
        if k < 0 || k > n { return Ok(Object::Integer(0)); }
        let mut res = 1i64;
        for i in 0..k {
            res *= n - i;
        }
        Ok(Object::Integer(res))
    });
    env.borrow_mut().set("perm".to_string(), perm_fn);

    let dict_fromkeys_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 { return Err("dict_fromkeys() takes 1 or 2 arguments (keys, [value])".to_string()); }
        let keys = match &args[0] {
            Object::List(l) => l.borrow().clone(),
            _ => return Err("dict_fromkeys() keys must be a list".to_string()),
        };
        let value = if args.len() == 2 { args[1].clone() } else { Object::Null };
        let mut d = HashMap::new();
        for key in keys {
            d.insert(key.clone(), value.clone());
        }
        Ok(Object::Dictionary(Rc::new(RefCell::new(d))))
    });
    env.borrow_mut().set("dict_fromkeys".to_string(), dict_fromkeys_fn);

    let isalnum_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("isalnum() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| c.is_alphanumeric()))),
            _ => Err(format!("isalnum() arg must be string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isalnum".to_string(), isalnum_fn);

    let islower_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("islower() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| !c.is_uppercase()) && s.chars().any(|c| c.is_lowercase()))),
            _ => Err(format!("islower() arg must be string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("islower".to_string(), islower_fn);

    let isupper_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("isupper() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| !c.is_lowercase()) && s.chars().any(|c| c.is_uppercase()))),
            _ => Err(format!("isupper() arg must be string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isupper".to_string(), isupper_fn);

    // Registry continues below...

    // Constants
    env.borrow_mut().set("true".to_string(), Object::Boolean(true));
    env.borrow_mut().set("false".to_string(), Object::Boolean(false));
    env.borrow_mut().set("null".to_string(), Object::Null);
    env.borrow_mut().set("True".to_string(), Object::Boolean(true));
    env.borrow_mut().set("False".to_string(), Object::Boolean(false));
    env.borrow_mut().set("None".to_string(), Object::Null);
    
    // Math constants
    env.borrow_mut().set("PI".to_string(), Object::Float(std::f64::consts::PI));
    env.borrow_mut().set("pi".to_string(), Object::Float(std::f64::consts::PI));
    env.borrow_mut().set("E".to_string(), Object::Float(std::f64::consts::E));
    env.borrow_mut().set("e".to_string(), Object::Float(std::f64::consts::E));
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let filename = &args[1];
        let contents = fs::read_to_string(filename).expect("Something went wrong reading the file");
        
        let lexer = Lexer::new(&contents);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program();
        
        if !parser.errors.is_empty() {
            for err in parser.errors {
                let flux_err = FluxError::new_parse(err.message, err.span);
                let report = Report::new(flux_err).with_source_code(contents.clone());
                println!("{:?}", report);
            }
        } else {
            let mut type_checker = crate::type_checker::TypeChecker::new();
            if let Err(e) = type_checker.check(&program) {
                let report = Report::new(e).with_source_code(contents.clone());
                println!("{:?}", report);
            } else {
                let env = Rc::new(RefCell::new(Environment::new()));
                register_builtins(env.clone());
                let mut interpreter = Interpreter::new();
                for stmt in program {
                    match interpreter.eval(stmt, env.clone()) {
                        Ok(_) => {},
                        Err(e) => {
                            let report = Report::new(e).with_source_code(contents.clone());
                            println!("{:?}", report);
                            break;
                        },
                    }
                }
            }
        }
    } else {
        println!("Flux Language (Pre-Alpha)");
        println!("Type 'exit' to quit.");
        
        let env = Rc::new(RefCell::new(Environment::new()));
        register_builtins(env.clone());
        let mut interpreter = Interpreter::new();
        let mut type_checker = crate::type_checker::TypeChecker::new();

        loop {
            print!(">> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            if input.trim() == "exit" {
                break;
            }
            
            let lexer = Lexer::new(&input);
            let mut parser = Parser::new(lexer);
            let program = parser.parse_program();
            
            if !parser.errors.is_empty() {
                for err in parser.errors {
                    let flux_err = FluxError::new_parse(err.message, err.span);
                    let report = Report::new(flux_err).with_source_code(input.clone());
                    println!("{:?}", report);
                }
            } else {
                if let Err(e) = type_checker.check(&program) {
                    let report = Report::new(e).with_source_code(input.clone());
                    println!("{:?}", report);
                } else {
                    for stmt in program {
                        match interpreter.eval(stmt, env.clone()) {
                            Ok(obj) => {
                                 if obj != Object::Null {
                                     println!("{}", obj);
                                 }
                            },
                            Err(e) => {
                                let report = Report::new(e).with_source_code(input.clone());
                                println!("{:?}", report);
                            },
                        }
                    }
                }
            }
        }
    }
}
