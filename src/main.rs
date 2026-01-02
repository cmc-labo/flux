mod token;
mod lexer;
mod ast;
mod parser;
mod object;
mod environment;
mod interpreter;
mod tensor;

use lexer::Lexer;
use parser::Parser;
use interpreter::Interpreter;
use environment::Environment;
use object::Object;
use pyo3::types::PyAnyMethods;
use std::io::{self, Write};
use std::env;
use std::fs;
use std::rc::Rc;
use std::cell::RefCell;

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
                pyo3::Python::with_gil(|py| {
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
        Ok(Object::List(elements))
    });
    env.borrow_mut().set("range".to_string(), range_fn);
    
    let len_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("len() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => Ok(Object::Integer(l.len() as i64)),
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
                pyo3::Python::with_gil(|py| {
                    let len = py_obj.bind(py).len()
                        .map_err(|e| format!("Python len() error: {}", e))?;
                    Ok(Object::Integer(len as i64))
                })
            },
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
        }
    });
    env.borrow_mut().set("type".to_string(), type_fn);

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
            Object::List(l) => {
                let mut total_int = 0;
                let mut total_float = 0.0;
                let mut has_float = false;

                for item in l {
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
            _ => Err(format!("sum() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("sum".to_string(), sum_fn);

    let min_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("min() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                if l.is_empty() { return Err("min() arg is an empty sequence".to_string()); }
                let mut min_val = None;
                for item in l {
                    let val = match item {
                        Object::Integer(i) => *i as f64,
                        Object::Float(f) => *f,
                        _ => return Err(format!("min() encountered non-numeric element: {}", item)),
                    };
                    if min_val.is_none() || val < min_val.unwrap() {
                        min_val = Some(val);
                    }
                }
                // Return same type as found if possible? Or just float?
                // For simplicity, if everything was int, return int.
                let mut all_int = true;
                let mut min_i = 0;
                let mut min_f = 0.0;
                let mut first = true;
                for item in l {
                    match item {
                        Object::Integer(i) => {
                            if first || (*i as f64) < (if all_int { min_i as f64 } else { min_f }) {
                                min_i = *i;
                                if !all_int { min_f = *i as f64; }
                            }
                        },
                        Object::Float(f) => {
                            if all_int {
                                all_int = false;
                                min_f = min_i as f64;
                            }
                            if first || *f < min_f {
                                min_f = *f;
                            }
                        },
                        _ => unreachable!(),
                    }
                    first = false;
                }
                if all_int { Ok(Object::Integer(min_i)) } else { Ok(Object::Float(min_f)) }
            },
            _ => Err(format!("min() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("min".to_string(), min_fn);

    let max_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("max() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                if l.is_empty() { return Err("max() arg is an empty sequence".to_string()); }
                let mut all_int = true;
                let mut max_i = 0;
                let mut max_f = 0.0;
                let mut first = true;
                for item in l {
                    match item {
                        Object::Integer(i) => {
                            if first || (*i as f64) > (if all_int { max_i as f64 } else { max_f }) {
                                max_i = *i;
                                if !all_int { max_f = *i as f64; }
                            }
                        },
                        Object::Float(f) => {
                            if all_int {
                                all_int = false;
                                max_f = max_i as f64;
                            }
                            if first || *f > max_f {
                                max_f = *f;
                            }
                        },
                        _ => return Err(format!("max() encountered non-numeric element: {}", item)),
                    }
                    first = false;
                }
                if all_int { Ok(Object::Integer(max_i)) } else { Ok(Object::Float(max_f)) }
            },
            _ => Err(format!("max() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("max".to_string(), max_fn);

    let mean_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("mean() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                if l.is_empty() { return Err("mean() arg is an empty sequence".to_string()); }
                let mut total = 0.0;
                for item in l {
                    match item {
                        Object::Integer(i) => total += *i as f64,
                        Object::Float(f) => total += f,
                        _ => return Err(format!("mean() encountered non-numeric element: {}", item)),
                    }
                }
                Ok(Object::Float(total / l.len() as f64))
            },
            _ => Err(format!("mean() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("mean".to_string(), mean_fn);

    let all_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("all() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                for item in l {
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
                for item in l {
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
                let mut rev_l = l.clone();
                rev_l.reverse();
                Ok(Object::List(rev_l))
            },
            _ => Err(format!("reverse() argument must be a list, got {}", args[0])),
        }
    });
    env.borrow_mut().set("reverse".to_string(), reverse_fn);

    let sorted_fn = Object::NativeFn(|args| {
        if args.len() != 1 {
            return Err("sorted() takes exactly 1 argument".to_string());
        }
        match &args[0] {
            Object::List(l) => {
                let mut sorted_l = l.clone();
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
                Ok(Object::List(sorted_l))
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
        Ok(Object::List(items))
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
        let result = list.iter().map(|obj| obj.to_string()).collect::<Vec<String>>().join(sep);
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
        let zipped: Vec<Object> = l1.iter().zip(l2.iter())
            .map(|(a, b)| Object::List(vec![a.clone(), b.clone()]))
            .collect();
        Ok(Object::List(zipped))
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
        let enumerated: Vec<Object> = l.iter().enumerate()
            .map(|(i, el)| Object::List(vec![Object::Integer(i as i64), el.clone()]))
            .collect();
        Ok(Object::List(enumerated))
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
            return Err("count() takes exactly 2 arguments (list, value)".to_string());
        }
        let list = match &args[0] {
            Object::List(val) => val,
            _ => return Err(format!("count() first argument must be a list, got {}", args[0])),
        };
        let target = &args[1];
        let c = list.iter().filter(|&item| item == target).count();
        Ok(Object::Integer(c as i64))
    });
    env.borrow_mut().set("count".to_string(), count_fn);

    // Constants
    env.borrow_mut().set("pi".to_string(), Object::Float(std::f64::consts::PI));
    env.borrow_mut().set("e".to_string(), Object::Float(std::f64::consts::E));

    // Constants
    env.borrow_mut().set("true".to_string(), Object::Boolean(true));
    env.borrow_mut().set("false".to_string(), Object::Boolean(false));
    env.borrow_mut().set("null".to_string(), Object::Null);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        let filename = &args[1];
        let input = fs::read_to_string(filename).expect("Something went wrong reading the file");
        let lexer = Lexer::new(&input);
        let mut parser = Parser::new(lexer);
        let program = parser.parse_program();
        
        if !parser.errors.is_empty() {
            println!("Parser errors:");
            for msg in parser.errors {
                println!("\t{}", msg);
            }
        } else {
            let env = Rc::new(RefCell::new(Environment::new()));
            register_builtins(env.clone());
            
            let mut interpreter = Interpreter::new();
            for stmt in program {
                match interpreter.eval(stmt, env.clone()) {
                    Ok(_) => {},
                    Err(e) => println!("Runtime Error: {}", e),
                }
            }
        }
    } else {
        println!("Flux Language (Pre-Alpha)");
        println!("Type 'exit' to quit.");
        
        let env = Rc::new(RefCell::new(Environment::new()));
        register_builtins(env.clone());

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
                println!("Parser errors:");
                for msg in parser.errors {
                    println!("\t{}", msg);
                }
            } else {
                let mut interpreter = Interpreter::new();
                for stmt in program {
                    match interpreter.eval(stmt, env.clone()) {
                        Ok(obj) => println!("{}", obj),
                        Err(e) => println!("Runtime Error: {}", e),
                    }
                }
            }
        }
    }
}
