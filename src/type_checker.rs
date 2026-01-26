use crate::ast::{Statement, StatementKind, Expression, ExpressionKind, Type, InfixOperator, Block};
use crate::error::FluxError;
use std::collections::HashMap;

#[derive(Clone)]
pub struct TypeEnv {
    vars: HashMap<String, Type>,
    parent: Option<Box<TypeEnv>>,
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            vars: HashMap::new(),
            parent: None,
        }
    }

    pub fn extend(parent: TypeEnv) -> Self {
        TypeEnv {
            vars: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    pub fn set(&mut self, name: String, ty: Type) {
        self.vars.insert(name, ty);
    }

    pub fn get(&self, name: &str) -> Option<Type> {
        if let Some(ty) = self.vars.get(name) {
            Some(ty.clone())
        } else if let Some(parent) = &self.parent {
            parent.get(name)
        } else {
            None
        }
    }
}

pub struct TypeChecker {
    env: TypeEnv,
    current_return_type: Option<Type>,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            env: Self::default_env(),
            current_return_type: None,
        }
    }

    fn default_env() -> TypeEnv {
        let mut env = TypeEnv::new();
        env.set("len".to_string(), Type::Function(vec![Type::Any], Box::new(Type::Int)));
        env.set("print".to_string(), Type::Function(vec![Type::Any], Box::new(Type::Null)));
        env.set("type".to_string(), Type::Function(vec![Type::Any], Box::new(Type::String)));
        env.set("range".to_string(), Type::Function(vec![Type::Int], Box::new(Type::List(Box::new(Type::Int)))));
        env.set("tensor".to_string(), Type::Function(vec![Type::Int], Box::new(Type::Tensor)));
        env.set("zeros".to_string(), Type::Function(vec![Type::List(Box::new(Type::Int))], Box::new(Type::Tensor)));
        env.set("ones".to_string(), Type::Function(vec![Type::List(Box::new(Type::Int))], Box::new(Type::Tensor)));
        env.set("rand".to_string(), Type::Function(vec![Type::List(Box::new(Type::Int))], Box::new(Type::Tensor)));
        env.set("sum".to_string(), Type::Function(vec![Type::Any], Box::new(Type::Float)));
        env.set("mean".to_string(), Type::Function(vec![Type::Any], Box::new(Type::Float)));
        env.set("keys".to_string(), Type::Function(vec![Type::Dictionary(Box::new(Type::Any), Box::new(Type::Any))], Box::new(Type::List(Box::new(Type::Any)))));
        env.set("values".to_string(), Type::Function(vec![Type::Dictionary(Box::new(Type::Any), Box::new(Type::Any))], Box::new(Type::List(Box::new(Type::Any)))));
        env.set("items".to_string(), Type::Function(vec![Type::Dictionary(Box::new(Type::Any), Box::new(Type::Any))], Box::new(Type::List(Box::new(Type::List(Box::new(Type::Any)))))));
        env.set("get".to_string(), Type::Function(vec![Type::Dictionary(Box::new(Type::Any), Box::new(Type::Any)), Type::Any, Type::Any], Box::new(Type::Any)));
        env.set("lstrip".to_string(), Type::Function(vec![Type::String], Box::new(Type::String)));
        env.set("rstrip".to_string(), Type::Function(vec![Type::String], Box::new(Type::String)));
        env.set("unique".to_string(), Type::Function(vec![Type::List(Box::new(Type::Any))], Box::new(Type::List(Box::new(Type::Any)))));
        env.set("dot".to_string(), Type::Function(vec![Type::Tensor, Type::Tensor], Box::new(Type::Tensor)));
        env.set("input".to_string(), Type::Function(vec![Type::Any], Box::new(Type::String)));
        env.set("sorted".to_string(), Type::Function(vec![Type::List(Box::new(Type::Any))], Box::new(Type::List(Box::new(Type::Any)))));
        env
    }

    pub fn check(&mut self, statements: &[Statement]) -> Result<(), FluxError> {
        let mut env = std::mem::replace(&mut self.env, TypeEnv::new());
        let result = (|| {
            for stmt in statements {
                self.check_statement(stmt, &mut env)?;
            }
            Ok(())
        })();
        self.env = env;
        result
    }

    fn check_statement(&mut self, stmt: &Statement, env: &mut TypeEnv) -> Result<(), FluxError> {
        match &stmt.kind {
            StatementKind::Let { name, value, type_hint } => {
                let inferred = self.infer_type(value, env)?;
                if let Some(hint) = type_hint {
                    if !self.is_compatible(hint, &inferred) {
                        return Err(FluxError::new_type(
                            format!("Type mismatch for '{}': declared {:?}, but got {:?}", name, hint, inferred),
                            stmt.span
                        ));
                    }
                    env.set(name.clone(), hint.clone());
                } else {
                    env.set(name.clone(), inferred);
                }
            }
            StatementKind::FunctionDef { name, params, body, return_type } => {
                let ret_ty = return_type.clone().unwrap_or(Type::Any);
                let func_ty = Type::Function(
                    params.iter().map(|(_, t)| t.clone().unwrap_or(Type::Any)).collect(),
                    Box::new(ret_ty.clone())
                );
                env.set(name.clone(), func_ty);

                let mut sub_env = TypeEnv::extend(env.clone());
                for (p_name, p_type) in params {
                    sub_env.set(p_name.clone(), p_type.clone().unwrap_or(Type::Any));
                }

                let old_ret = self.current_return_type.take();
                self.current_return_type = Some(ret_ty);
                self.check_block(body, &mut sub_env)?;
                self.current_return_type = old_ret;
            }
            StatementKind::Return(expr_opt) => {
                let ret_ty = match expr_opt {
                    Some(expr) => self.infer_type(expr, env)?,
                    None => Type::Null,
                };
                if let Some(expected) = &self.current_return_type {
                    if !self.is_compatible(expected, &ret_ty) {
                        return Err(FluxError::new_type(
                            format!("Return type mismatch: expected {:?}, but got {:?}", expected, ret_ty),
                            stmt.span
                        ));
                    }
                }
            }
            StatementKind::If { condition, consequence, elif_branches, alternative } => {
                let _ = self.infer_type(condition, env)?;
                // condition should be truthy, almost anything works in Flux but maybe warn?
                self.check_block(consequence, &mut TypeEnv::extend(env.clone()))?;
                for (elif_cond, elif_body) in elif_branches {
                    self.infer_type(elif_cond, env)?;
                    self.check_block(elif_body, &mut TypeEnv::extend(env.clone()))?;
                }
                if let Some(alt) = alternative {
                    self.check_block(alt, &mut TypeEnv::extend(env.clone()))?;
                }
            }
            StatementKind::While { condition, body } => {
                self.infer_type(condition, env)?;
                self.check_block(body, &mut TypeEnv::extend(env.clone()))?;
            }
            StatementKind::For { variable, iterable, body } => {
                let iter_ty = self.infer_type(iterable, env)?;
                let elem_ty = match iter_ty {
                    Type::List(inner) => *inner,
                    Type::Tensor => Type::Float,
                    Type::String => Type::String,
                    _ => Type::Any,
                };
                let mut sub_env = TypeEnv::extend(env.clone());
                sub_env.set(variable.clone(), elem_ty);
                self.check_block(body, &mut sub_env)?;
            }
            StatementKind::IndexAssign { object, index, value } => {
                let obj_ty = self.infer_type(object, env)?;
                let idx_ty = self.infer_type(index, env)?;
                let val_ty = self.infer_type(value, env)?;

                match obj_ty {
                    Type::List(inner) => {
                        if idx_ty != Type::Int {
                            return Err(FluxError::new_type(format!("List index must be int, got {:?}", idx_ty), stmt.span));
                        }
                        if !self.is_compatible(&inner, &val_ty) {
                            return Err(FluxError::new_type(format!("Cannot assign {:?} to list of {:?}", val_ty, inner), stmt.span));
                        }
                    }
                    Type::Dictionary(k, v) => {
                        if !self.is_compatible(&k, &idx_ty) {
                            return Err(FluxError::new_type(format!("Dictionary key mismatch: expected {:?}, got {:?}", k, idx_ty), stmt.span));
                        }
                        if !self.is_compatible(&v, &val_ty) {
                            return Err(FluxError::new_type(format!("Dictionary value mismatch: expected {:?}, got {:?}", v, val_ty), stmt.span));
                        }
                    }
                    Type::Tensor => {
                        // Tensors usually support slicing/indexing but for now assume float assign
                    }
                    _ => {}
                }
            }
            StatementKind::Import { path, alias } => {
                let name = alias.clone().unwrap_or_else(|| {
                    path.split('.').last().unwrap_or(path).to_string()
                });
                env.set(name, Type::Any);
            }
            StatementKind::Expression(expr) => {
                self.infer_type(expr, env)?;
            }
            StatementKind::Print(expressions) => {
                for expr in expressions {
                    self.infer_type(expr, env)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn check_block(&mut self, block: &Block, env: &mut TypeEnv) -> Result<(), FluxError> {
        for stmt in &block.statements {
            self.check_statement(stmt, env)?;
        }
        Ok(())
    }

    fn infer_type(&self, expr: &Expression, env: &TypeEnv) -> Result<Type, FluxError> {
        match &expr.kind {
            ExpressionKind::Integer(_) => Ok(Type::Int),
            ExpressionKind::Float(_) => Ok(Type::Float),
            ExpressionKind::String(_) => Ok(Type::String),
            ExpressionKind::Identifier(name) => {
                env.get(name).ok_or_else(|| FluxError::new_type(format!("Undefined variable: {}", name), expr.span))
            }
            ExpressionKind::List(elements) => {
                if elements.is_empty() {
                    Ok(Type::List(Box::new(Type::Any)))
                } else {
                    let mut common_ty = self.infer_type(&elements[0], env)?;
                    for el in &elements[1..] {
                        let el_ty = self.infer_type(el, env)?;
                        if el_ty != common_ty {
                            // If elements differ, fallback to Any for now or handle union
                            common_ty = Type::Any;
                            break;
                        }
                    }
                    Ok(Type::List(Box::new(common_ty)))
                }
            }
            ExpressionKind::Infix { left, operator, right } => {
                let l_ty = self.infer_type(left, env)?;
                let r_ty = self.infer_type(right, env)?;
                match operator {
                    InfixOperator::Plus | InfixOperator::Minus | InfixOperator::Multiply | InfixOperator::Divide | InfixOperator::Modulo => {
                        if l_ty == Type::Int && r_ty == Type::Int { Ok(Type::Int) }
                        else if (l_ty == Type::Int || l_ty == Type::Float) && (r_ty == Type::Int || r_ty == Type::Float) { Ok(Type::Float) }
                        else if l_ty == Type::Tensor || r_ty == Type::Tensor { Ok(Type::Tensor) }
                        else { Ok(Type::Any) }
                    }
                    InfixOperator::Equal | InfixOperator::NotEqual | InfixOperator::LessThan | InfixOperator::GreaterThan |
                    InfixOperator::LessThanOrEqual | InfixOperator::GreaterThanOrEqual | InfixOperator::And | InfixOperator::Or |
                    InfixOperator::In | InfixOperator::NotIn => Ok(Type::Bool),
                    InfixOperator::Power => Ok(Type::Float),
                    InfixOperator::MatrixMultiply => Ok(Type::Tensor),
                    InfixOperator::BitwiseAnd | InfixOperator::BitwiseOr | InfixOperator::BitwiseXor |
                    InfixOperator::ShiftLeft | InfixOperator::ShiftRight => {
                        if l_ty == Type::Set(Box::new(Type::Any)) || r_ty == Type::Set(Box::new(Type::Any)) {
                            // If any is set, result is set
                            // Better check if both are sets or set-compatible
                            Ok(l_ty) 
                        } else {
                            Ok(Type::Int)
                        }
                    },
                }
            }
            ExpressionKind::Prefix { operator, right } => {
                let r_ty = self.infer_type(right, env)?;
                match operator {
                    crate::ast::PrefixOperator::Minus => Ok(r_ty),
                    crate::ast::PrefixOperator::Not => Ok(Type::Bool),
                    crate::ast::PrefixOperator::BitwiseNot => Ok(Type::Int),
                }
            }
            ExpressionKind::Call { function, arguments } => {
                let func_ty = self.infer_type(function, env)?;
                match func_ty {
                    Type::Function(params, ret) => {
                        // Check argument count
                        if arguments.len() != params.len() {
                             // Some functions might have varargs, but for now strict
                        }
                        Ok(*ret)
                    }
                    _ => Ok(Type::Any), // Might be a PyObject or NativeFn we didn't track well
                }
            }
            ExpressionKind::Index { object, index: _ } => {
                let obj_ty = self.infer_type(object, env)?;
                match obj_ty {
                    Type::List(inner) => Ok(*inner),
                    Type::Dictionary(_, val) => Ok(*val),
                    Type::String => Ok(Type::String),
                    Type::Tensor => Ok(Type::Float), // Tensor index returns float for now
                    _ => Ok(Type::Any),
                }
            }
            ExpressionKind::Dictionary(pairs) => {
                let mut key_type = Type::Any;
                let mut val_type = Type::Any;
                if !pairs.is_empty() {
                    let (k, v) = &pairs[0];
                    key_type = self.infer_type(k, env)?;
                    val_type = self.infer_type(v, env)?;
                }
                Ok(Type::Dictionary(Box::new(key_type), Box::new(val_type)))
            },
            ExpressionKind::Set(elements) => {
                if elements.is_empty() {
                    Ok(Type::Set(Box::new(Type::Any)))
                } else {
                    let mut common_ty = self.infer_type(&elements[0], env)?;
                    for el in &elements[1..] {
                        let el_ty = self.infer_type(el, env)?;
                        if el_ty != common_ty {
                            common_ty = Type::Any;
                            break;
                        }
                    }
                    Ok(Type::Set(Box::new(common_ty)))
                }
            },
            ExpressionKind::Slice { .. } => Ok(Type::Any), // Slices are internal/index only for now
            ExpressionKind::Get { object: _, name: _ } => {
                 // Attribute access. Hard to type without more info.
                 Ok(Type::Any)
            }
            ExpressionKind::ListComprehension { element, variable, iterable, condition } => {
                let iter_ty = self.infer_type(iterable, env)?;
                let elem_ty = match iter_ty {
                    Type::List(inner) => *inner,
                    Type::Tensor => Type::Float,
                    _ => Type::Any,
                };
                let mut sub_env = TypeEnv::extend(env.clone());
                sub_env.set(variable.clone(), elem_ty);
                
                if let Some(cond) = condition {
                    self.infer_type(cond, &sub_env)?;
                }
                
                let res_ty = self.infer_type(element, &sub_env)?;
                Ok(Type::List(Box::new(res_ty)))
            },
            ExpressionKind::MethodCall { object: _, method, arguments: _ } => {
                // Look up method as function
                match env.get(method) {
                    Some(Type::Function(_, ret)) => Ok(*ret),
                    _ => Ok(Type::Any),
                }
            },
            ExpressionKind::Lambda { params, body } => {
                let mut sub_env = TypeEnv::extend(env.clone());
                for p in params {
                    sub_env.set(p.clone(), Type::Any);
                }
                let ret_ty = self.infer_type(body, &sub_env)?;
                Ok(Type::Function(vec![Type::Any; params.len()], Box::new(ret_ty)))
            },
            ExpressionKind::FString(parts) => {
                for part in parts {
                    if let crate::ast::FStringPart::Expression(expr) = part {
                        self.infer_type(expr, env)?;
                    }
                }
                Ok(Type::String)
            },
            ExpressionKind::Ternary { condition, consequence, alternative } => {
                let _ = self.infer_type(condition, env)?;
                let t1 = self.infer_type(consequence, env)?;
                let t2 = self.infer_type(alternative, env)?;
                if t1 == t2 {
                    Ok(t1)
                } else if (t1 == Type::Int && t2 == Type::Float) || (t1 == Type::Float && t2 == Type::Int) {
                    Ok(Type::Float)
                } else {
                    Ok(Type::Any)
                }
            },
        }
    }

    fn is_compatible(&self, expected: &Type, actual: &Type) -> bool {
        if expected == &Type::Any || actual == &Type::Any {
            return true;
        }
        if expected == actual {
            return true;
        }
        // int is compatible with float (automatic coercion)
        if expected == &Type::Float && actual == &Type::Int {
            return true;
        }
        // Handle List compatibility
        if let (Type::List(e_inner), Type::List(a_inner)) = (expected, actual) {
            return self.is_compatible(e_inner, a_inner);
        }
        // Handle Dictionary compatibility
        if let (Type::Dictionary(e_k, e_v), Type::Dictionary(a_k, a_v)) = (expected, actual) {
            return self.is_compatible(e_k, a_k) && self.is_compatible(e_v, a_v);
        }
        // Handle Set compatibility
        if let (Type::Set(e_inner), Type::Set(a_inner)) = (expected, actual) {
            return self.is_compatible(e_inner, a_inner);
        }
        false
    }
}
