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
use token::Token;
use interpreter::Interpreter;
use environment::Environment;
use object::Object;
use std::io::{self, Write};
use std::env;
use std::fs;
use std::rc::Rc;
use std::cell::RefCell;

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
            
            // Register built-in functions
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
