
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
use std::cell::RefCell;
use std::time::{SystemTime, UNIX_EPOCH};
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
            _ => Err("tensor() argument must be an integer".to_string()),
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
            Object::Module { .. } => Ok(Object::String("module".to_string())),
            Object::Slice { .. } => Ok(Object::String("slice".to_string())),
        }
    });
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

    let transpose_fn = Object::NativeFn(|args| {
        if args.len() != 1 { return Err("transpose() takes exactly 1 argument".to_string()); }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Tensor(t.transpose())),
            _ => Err("transpose() argument must be a tensor".to_string()),
        }
    });
    env.borrow_mut().set("transpose".to_string(), transpose_fn);

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
            _ => Err(format!("abs() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("abs".to_string(), abs_fn);

    let sum_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sum() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Tensor(t) => Ok(Object::Float(t.sum())),
            Object::List(l) => {
                let mut total_int = 0;
                let mut total_float = 0.0;
                let mut has_float = false;

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
            _ => Err(format!("sum() argument must be a list or tensor, got {}", args[0])),
        }
    });
    env.borrow_mut().set("sum".to_string(), sum_fn);

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

    let min_fn = Object::NativeFn(|args| {
        if args.len() == 0 {
            return Err("min() expects at least 1 argument".to_string());
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
        let mut min_i = 0;
        let mut min_f = 0.0;
        let mut first = true;
        for item in elements {
            match item {
                Object::Integer(i) => {
                    if first || (i as f64) < (if all_int { min_i as f64 } else { min_f }) {
                        min_i = i;
                        if !all_int { min_f = i as f64; }
                    }
                },
                Object::Float(f) => {
                    if all_int {
                        all_int = false;
                        min_f = min_i as f64;
                    }
                    if first || f < min_f {
                        min_f = f;
                    }
                },
                _ => return Err(format!("min() encountered non-numeric element: {}", item)),
            }
            first = false;
        }
        if all_int { Ok(Object::Integer(min_i)) } else { Ok(Object::Float(min_f)) }
    });
    env.borrow_mut().set("min".to_string(), min_fn);

    let max_fn = Object::NativeFn(|args| {
        if args.len() == 0 {
            return Err("max() expects at least 1 argument".to_string());
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
        let mut max_i = 0;
        let mut max_f = 0.0;
        let mut first = true;
        for item in elements {
            match item {
                Object::Integer(i) => {
                    if first || (i as f64) > (if all_int { max_i as f64 } else { max_f }) {
                        max_i = i;
                        if !all_int { max_f = i as f64; }
                    }
                },
                Object::Float(f) => {
                    if all_int {
                        all_int = false;
                        max_f = max_i as f64;
                    }
                    if first || f > max_f {
                        max_f = f;
                    }
                },
                _ => return Err(format!("max() encountered non-numeric element: {}", item)),
            }
            first = false;
        }
        if all_int { Ok(Object::Integer(max_i)) } else { Ok(Object::Float(max_f)) }
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
            _ => Err(format!("all() argument must be a list, got {}", args[0])),
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
            _ => Err(format!("any() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("any".to_string(), any_fn);

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

    let round_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("round() takes 1 or 2 arguments".to_string());
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
        if args.len() != 1 {
            return Err("strip() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.trim().to_string())),
            _ => Err(format!("strip() argument must be a string, got {}", args[0])),
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
        if args.len() < 1 || args.len() > 2 {
            return Err("split() takes 1 or 2 arguments".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("split() first argument must be a string, got {}", args[0])),
        };
        let sep = if args.len() == 2 {
            match &args[1] {
                Object::String(val) => val,
                _ => return Err(format!("split() separator must be a string, got {}", args[1])),
            }
        } else {
            " "
        };
        let items: Vec<Object> = if sep.is_empty() {
            s.chars().map(|c| Object::String(c.to_string())).collect()
        } else {
            s.split(sep).map(|item| Object::String(item.to_string())).collect()
        };
        Ok(Object::List(Rc::new(RefCell::new(items))))
    });
    env.borrow_mut().set("split".to_string(), split_fn);

    let join_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("join() takes exactly 2 arguments (list, separator)".to_string());
        }
        let list = match &args[0] {
            Object::List(l) => l,
            _ => return Err(format!("join() first argument must be a list, got {}", args[0])),
        };
        let sep = match &args[1] {
            Object::String(s) => s,
            _ => return Err(format!("join() second argument must be a string, got {}", args[1])),
        };
        let result = list.borrow().iter().map(|obj| obj.to_string()).collect::<Vec<String>>().join(sep);
        Ok(Object::String(result))
    });
    env.borrow_mut().set("join".to_string(), join_fn);

    let replace_fn = Object::NativeFn(|args| {
        if args.len() != 3 {
            return Err("replace() takes exactly 3 arguments (string, old, new)".to_string());
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
        Ok(Object::String(s.replace(old, new)))
    });
    env.borrow_mut().set("replace".to_string(), replace_fn);

    let find_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("find() takes exactly 2 arguments (string, substring)".to_string());
        }
        let s = match &args[0] {
            Object::String(val) => val,
            _ => return Err(format!("find() first argument must be a string, got {}", args[0])),
        };
        let sub = match &args[1] {
            Object::String(val) => val,
            _ => return Err(format!("find() second argument must be a string, got {}", args[1])),
        };
        match s.find(sub) {
            Some(idx) => Ok(Object::Integer(idx as i64)),
            None => Ok(Object::Integer(-1)),
        }
    });
    env.borrow_mut().set("find".to_string(), find_fn);

    let floor_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("floor() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::Integer(i) => Ok(Object::Integer(*i)),
            Object::Float(f) => Ok(Object::Integer(f.floor() as i64)),
            _ => Err(format!("floor() not supported for {}", args[0])),
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
            _ => Err(format!("ceil() not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("ceil".to_string(), ceil_fn);

    let sqrt_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sqrt() takes exactly 1 argument".to_string());
        }
        let val = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("sqrt() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(val.sqrt()))
    });
    env.borrow_mut().set("sqrt".to_string(), sqrt_fn);

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
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("exp() argument must be numeric, got {}", args[0])),
        };
        Ok(Object::Float(val.exp()))
    });
    env.borrow_mut().set("exp".to_string(), exp_fn);

    let log_fn = Object::NativeFn(|args| {
        if args.len() < 1 || args.len() > 2 {
            return Err("log() takes 1 or 2 arguments (x, [base])".to_string());
        }
        let x = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err(format!("log() first argument must be numeric, got {}", args[0])),
        };
        let base = if args.len() == 2 {
            match &args[1] {
                Object::Integer(i) => *i as f64,
                Object::Float(f) => *f,
                _ => return Err(format!("log() base must be numeric, got {}", args[1])),
            }
        } else {
            std::f64::consts::E
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

    let zip_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("zip() takes exactly 2 arguments (list1, list2)".to_string());
        }
        let l1 = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("zip() first argument must be a list, got {}", args[0])),
        };
        let l2 = match &args[1] {
            Object::List(val) => val,
            _ => return Err(format!("zip() second argument must be a list, got {}", args[1])),
        };
        let zipped: Vec<Object> = l1.borrow().iter().zip(l2.borrow().iter())
            .map(|(a, b)| Object::List(Rc::new(RefCell::new(vec![a.clone(), b.clone()]))))
            .collect();
        Ok(Object::List(Rc::new(RefCell::new(zipped))))
    });
    env.borrow_mut().set("zip".to_string(), zip_fn);

    let enumerate_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("enumerate() takes exactly 1 argument (list)".to_string());
        }
        let l = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("enumerate() argument must be a list, got {}", args[0])),
        };
        let enumerated: Vec<Object> = l.borrow().iter().enumerate()
            .map(|(i, el)| Object::List(Rc::new(RefCell::new(vec![Object::Integer(i as i64), el.clone()]))))
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

    let isalpha_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("isalpha() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::Boolean(!s.is_empty() && s.chars().all(|c| c.is_alphabetic()))),
            _ => Err(format!("isalpha() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("isalpha".to_string(), isalpha_fn);

    let count_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("count() takes exactly 2 arguments".to_string());
        }
        match &args[0] {
            Object::List(list) => {
                let target = &args[1];
                let count = list.borrow().iter().filter(|x| *x == target).count();
                Ok(Object::Integer(count as i64))
            },
            Object::String(s) => {
                let sub = match &args[1] {
                    Object::String(val) => val,
                    _ => return Err(format!("count() second argument must be a string for string search, got {}", args[1])),
                };
                if sub.is_empty() {
                    return Ok(Object::Integer((s.len() + 1) as i64));
                }
                let c = s.matches(sub).count();
                Ok(Object::Integer(c as i64))
            },
            _ => Err(format!("count() first argument must be a list or string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("count".to_string(), count_fn);

    let input_fn = Object::NativeFn(|args| {
        let msg = if args.len() == 1 {
            args[0].to_string()
        } else {
            "".to_string()
        };
        print!("{}", msg);
        io::stdout().flush().unwrap();
        let mut response = String::new();
        io::stdin().read_line(&mut response).unwrap();
        Ok(Object::String(response.trim_end().to_string()))
    });
    env.borrow_mut().set("input".to_string(), input_fn);

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
            _ => Err(format!("list() conversion not supported for {}", args[0])),
        }
    });
    env.borrow_mut().set("list".to_string(), list_conv_fn);

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
        let index = match &args[1] {
            Object::Integer(i) => *i as usize,
            _ => return Err(format!("insert() second argument must be an integer, got {}", args[1])),
        };
        let item = args[2].clone();
        let list_len = list.borrow().len();
        if index > list_len {
            return Err(format!("insert() index {} out of range for list of length {}", index, list_len));
        }
        list.borrow_mut().insert(index, item);
        Ok(args[0].clone())  // Return the list itself
    });
    env.borrow_mut().set("insert".to_string(), insert_fn);

    let capitalize_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("capitalize() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => {
                if s.is_empty() {
                    return Ok(Object::String("".to_string()));
                }
                let mut c = s.chars();
                let first = c.next().unwrap().to_uppercase().to_string();
                let rest = c.as_str().to_lowercase();
                Ok(Object::String(format!("{}{}", first, rest)))
            },
            _ => Err(format!("capitalize() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("capitalize".to_string(), capitalize_fn);

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
            _ => Err(format!("pop() first argument must be a list or dictionary, got {}", args[0])),
        }
    });
    env.borrow_mut().set("pop".to_string(), pop_fn.clone());
    env.borrow_mut().set("pop_at".to_string(), pop_fn);

    let remove_val_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("remove() takes exactly 2 arguments (list, item)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("remove() first argument must be a list, got {}", args[0])),
        };
        let target = &args[1];
        let pos = list.borrow().iter().position(|x| x == target);
        if let Some(idx) = pos {
            list.borrow_mut().remove(idx);
        }
        Ok(args[0].clone())  // Return the list itself
    });
    env.borrow_mut().set("remove".to_string(), remove_val_fn);

    let index_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("index() takes exactly 2 arguments (list, item)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("index() first argument must be a list, got {}", args[0])),
        };
        let target = &args[1];
        let pos = list.borrow().iter().position(|x| x == target);
        match pos {
            Some(p) => Ok(Object::Integer(p as i64)),
            None => Ok(Object::Integer(-1)),
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
        while b > 0 {
            a %= b;
            std::mem::swap(&mut a, &mut b);
        }
        Ok(Object::Integer(a))
    });
    env.borrow_mut().set("gcd".to_string(), gcd_fn);

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

    let hex_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("hex() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("hex() argument must be an integer, got {}", args[0])),
        };
        Ok(Object::String(format!("0x{:x}", n)))
    });
    env.borrow_mut().set("hex".to_string(), hex_fn);

    let oct_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("oct() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("oct() argument must be an integer, got {}", args[0])),
        };
        Ok(Object::String(format!("0o{:o}", n)))
    });
    env.borrow_mut().set("oct".to_string(), oct_fn);

    let bin_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("bin() takes exactly 1 argument".to_string());
        }
        let n = match &args[0] {
            Object::Integer(i) => *i,
            _ => return Err(format!("bin() argument must be an integer, got {}", args[0])),
        };
        Ok(Object::String(format!("0b{:b}", n)))
    });
    env.borrow_mut().set("bin".to_string(), bin_fn);

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
        let padding = "0".repeat(width - s.len());
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
        if args.len() != 2 {
            return Err("hypot() takes exactly 2 arguments".to_string());
        }
        let x = match &args[0] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err("hypot() arguments must be numeric".to_string()),
        };
        let y = match &args[1] {
            Object::Integer(i) => *i as f64,
            Object::Float(f) => *f,
            _ => return Err("hypot() arguments must be numeric".to_string()),
        };
        Ok(Object::Float(x.hypot(y)))
    });
    env.borrow_mut().set("hypot".to_string(), hypot_fn);

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

    // Constants
    env.borrow_mut().set("pi".to_string(), Object::Float(std::f64::consts::PI));
    env.borrow_mut().set("e".to_string(), Object::Float(std::f64::consts::E));
    env.borrow_mut().set("PI".to_string(), Object::Float(std::f64::consts::PI));
    env.borrow_mut().set("E".to_string(), Object::Float(std::f64::consts::E));

    let keys_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("keys() takes exactly 1 argument".to_string());
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
            return Err("values() takes exactly 1 argument".to_string());
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
        if args.len() != 1 {
            return Err("items() takes exactly 1 argument".to_string());
        }
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
             return Err("get() takes 2 or 3 arguments (dict, key, [default])".to_string());
        }
        let dict = match &args[0] {
            Object::Dictionary(d) => d,
            _ => return Err(format!("get() first argument must be a dictionary, got {}", args[0])),
        };
        let key = &args[1];
        let default = if args.len() == 3 { args[2].clone() } else { Object::Null };
        
        match dict.borrow().get(key) {
            Some(val) => Ok(val.clone()),
            None => Ok(default),
        }
    });
    env.borrow_mut().set("get".to_string(), get_fn);

    let lstrip_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("lstrip() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.trim_start().to_string())),
            _ => Err(format!("lstrip() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("lstrip".to_string(), lstrip_fn);

    let rstrip_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("rstrip() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.trim_end().to_string())),
            _ => Err(format!("rstrip() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("rstrip".to_string(), rstrip_fn);
    
    // Combined strip function
    let strip_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("strip() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::String(s) => Ok(Object::String(s.trim().to_string())),
            _ => Err(format!("strip() argument must be a string, got {}", args[0])),
        }
    });
    env.borrow_mut().set("strip".to_string(), strip_fn);
    
    // clear function for lists and dicts
    let clear_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("clear() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                l.borrow_mut().clear();
                Ok(Object::Null)
            },
            Object::Dictionary(d) => {
                d.borrow_mut().clear();
                Ok(Object::Null)
            },
            _ => Err(format!("clear() argument must be a list or dictionary, got {}", args[0])),
        }
    });
    env.borrow_mut().set("clear".to_string(), clear_fn);
    
    // replace function for strings
    let replace_fn = Object::NativeFn(|args| {
        if args.len() != 3 {
            return Err("replace() takes exactly 3 arguments (string, old, new)".to_string());
        }
        match (&args[0], &args[1], &args[2]) {
            (Object::String(s), Object::String(old), Object::String(new)) => {
                Ok(Object::String(s.replace(old, new)))
            },
            _ => Err("replace() arguments must be strings".to_string()),
        }
    });
    env.borrow_mut().set("replace".to_string(), replace_fn);
    
    // startswith function for strings
    let startswith_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("startswith() takes exactly 2 arguments (string, prefix)".to_string());
        }
        match (&args[0], &args[1]) {
            (Object::String(s), Object::String(prefix)) => {
                Ok(Object::Boolean(s.starts_with(prefix)))
            },
            _ => Err("startswith() arguments must be strings".to_string()),
        }
    });
    env.borrow_mut().set("startswith".to_string(), startswith_fn);
    
    // endswith function for strings
    let endswith_fn = Object::NativeFn(|args| {
        if args.len() != 2 {
            return Err("endswith() takes exactly 2 arguments (string, suffix)".to_string());
        }
        match (&args[0], &args[1]) {
            (Object::String(s), Object::String(suffix)) => {
                Ok(Object::Boolean(s.ends_with(suffix)))
            },
            _ => Err("endswith() arguments must be strings".to_string()),
        }
    });
    env.borrow_mut().set("endswith".to_string(), endswith_fn);
    
    let unique_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("unique() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut unique_list = Vec::new();
                for item in l.borrow().iter() {
                    if !unique_list.contains(item) {
                       unique_list.push(item.clone());
                    }
                }
                Ok(Object::List(Rc::new(RefCell::new(unique_list))))
            },
             _ => Err(format!("unique() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("unique".to_string(), unique_fn);

    // Constants
    env.borrow_mut().set("true".to_string(), Object::Boolean(true));
    env.borrow_mut().set("false".to_string(), Object::Boolean(false));
    env.borrow_mut().set("null".to_string(), Object::Null);
    env.borrow_mut().set("True".to_string(), Object::Boolean(true));
    env.borrow_mut().set("False".to_string(), Object::Boolean(false));
    env.borrow_mut().set("None".to_string(), Object::Null);
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
