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
