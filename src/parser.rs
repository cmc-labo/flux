use crate::token::Token;
use crate::lexer::Lexer;
use crate::ast::{Statement, StatementKind, Expression, ExpressionKind, Block, InfixOperator, PrefixOperator};
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct ParserError {
    pub message: String,
    pub span: Span,
}

#[derive(PartialEq, PartialOrd)]
enum Precedence {
    Lowest,
    Ternary,     // a if condition else b
    LogicalOr,   // or
    LogicalAnd,  // and
    Equals,      // ==
    BitwiseOr,   // |
    BitwiseXor,  // ^
    BitwiseAnd,  // &
    Shift,       // << >>
    Sum,         // +
    Product,     // * / %
    Prefix,      // unary - or ! or ~
    Power,       // **
    Call,        // myFunction(X)
    Index,       // array[index] or object.property
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    cur_token: Token,
    cur_span: Span,
    peek_token: Token,
    peek_span: Span,
    pub errors: Vec<ParserError>,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut p = Parser {
            lexer,
            cur_token: Token::EOF,
            cur_span: Span::new(0, 0),
            peek_token: Token::EOF,
            peek_span: Span::new(0, 0),
            errors: Vec::new(),
        };
        // Read two tokens to initialize cur and peek
        p.next_token();
        p.next_token();
        p
    }

    fn next_token(&mut self) {
        self.cur_token = self.peek_token.clone();
        self.cur_span = self.peek_span;
        let (tok, span) = self.lexer.next_token();
        self.peek_token = tok;
        self.peek_span = span;
    }

    pub fn parse_program(&mut self) -> Vec<Statement> {
        let mut statements = Vec::new();
        while self.cur_token != Token::EOF {
            if let Some(stmt) = self.parse_statement() {
                statements.push(stmt);
            }
            self.next_token();
        }
        statements
    }

    fn parse_statement(&mut self) -> Option<Statement> {
        match self.cur_token {
            Token::Let => self.parse_let_statement(),
            Token::Return => self.parse_return_statement(),
            Token::Def => self.parse_function_def(),
            Token::If => self.parse_if_statement(),
            Token::While => self.parse_while_statement(),
            Token::For => self.parse_for_statement(),
            Token::Print => self.parse_print_statement(),
            Token::Import => self.parse_import_statement(),
            Token::Break => {
                let start = self.cur_span;
                self.next_token();
                if self.peek_token_is(&Token::Newline) { self.next_token(); }
                Some(Statement { kind: StatementKind::Break, span: start })
            }
            Token::Continue => {
                let start = self.cur_span;
                self.next_token();
                if self.peek_token_is(&Token::Newline) { self.next_token(); }
                Some(Statement { kind: StatementKind::Continue, span: start })
            }
            Token::Assert => self.parse_assert_statement(),
            Token::Newline => None, // Skip empty lines
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_assert_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip assert
        let condition = self.parse_expression(Precedence::Lowest)?;
        
        let mut message = None;
        if self.peek_token_is(&Token::Comma) {
            self.next_token(); // skip ,
            self.next_token(); // move to message expr
            message = self.parse_expression(Precedence::Lowest);
        }

        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        let end = if let Some(msg) = &message { msg.span } else { condition.span };

        Some(Statement {
            span: start.join(end),
            kind: StatementKind::Assert { condition, message },
        })
    }

    fn parse_type(&mut self) -> Option<crate::ast::Type> {
        let t = match self.cur_token {
            Token::Identifier(ref name) => {
                match name.as_str() {
                    "int" => crate::ast::Type::Int,
                    "float" => crate::ast::Type::Float,
                    "string" => crate::ast::Type::String,
                    "bool" => crate::ast::Type::Bool,
                    "null" | "None" => crate::ast::Type::Null,
                    "tensor" => crate::ast::Type::Tensor,
                    "any" => crate::ast::Type::Any,
                    "list" => {
                        if self.peek_token_is(&Token::LBracket) {
                            self.next_token(); // consume list
                            self.next_token(); // consume [
                            let inner = self.parse_type()?;
                            if !self.expect_peek(Token::RBracket) { return None; }
                            crate::ast::Type::List(Box::new(inner))
                        } else {
                            crate::ast::Type::List(Box::new(crate::ast::Type::Any))
                        }
                    },
                    "set" => {
                        if self.peek_token_is(&Token::LBracket) {
                            self.next_token(); // consume set
                            self.next_token(); // consume [
                            let inner = self.parse_type()?;
                            if !self.expect_peek(Token::RBracket) { return None; }
                            crate::ast::Type::Set(Box::new(inner))
                        } else {
                            crate::ast::Type::Set(Box::new(crate::ast::Type::Any))
                        }
                    },
                    "dict" => {
                        if self.peek_token_is(&Token::LBracket) {
                            self.next_token(); // consume dict
                            self.next_token(); // consume [
                            let key = self.parse_type()?;
                            if !self.expect_peek(Token::Comma) { return None; }
                            self.next_token();
                            let val = self.parse_type()?;
                            if !self.expect_peek(Token::RBracket) { return None; }
                            crate::ast::Type::Dictionary(Box::new(key), Box::new(val))
                        } else {
                            crate::ast::Type::Dictionary(Box::new(crate::ast::Type::Any), Box::new(crate::ast::Type::Any))
                        }
                    },
                    _ => {
                        self.errors.push(ParserError { message: format!("Unknown type: {}", name), span: self.cur_span });
                        return None;
                    }
                }
            },
             _ => {
                self.errors.push(ParserError { message: format!("Expected type identifier, got {:?}", self.cur_token), span: self.cur_span });
                return None;
            }
        };
        Some(t)
    }

    fn parse_let_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        if !self.expect_peek(Token::Identifier("".to_string())) {
            return None;
        }
        
        let mut names = Vec::new();
        match &self.cur_token {
            Token::Identifier(n) => names.push(n.clone()),
            _ => return None,
        };

        while self.peek_token_is(&Token::Comma) {
            self.next_token(); // ,
            if !self.expect_peek(Token::Identifier("".to_string())) {
                return None;
            }
            match &self.cur_token {
                Token::Identifier(n) => names.push(n.clone()),
                _ => return None,
            }
        }

        let mut type_hint = None;
        if names.len() == 1 && self.peek_token_is(&Token::Colon) {
            self.next_token(); // identifier
            self.next_token(); // :
            type_hint = self.parse_type();
        }

        if !self.expect_peek(Token::Assign) {
            return None;
        }

        self.next_token(); // skip =

        let value = self.parse_expression(Precedence::Lowest)?;

        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement {
            span: start.join(value.span),
            kind: StatementKind::Let { names, value, type_hint },
        })
    }

    fn parse_return_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip return

        let mut end = start;
        let return_value = if self.cur_token_is(&Token::Newline) || self.cur_token_is(&Token::EOF) {
            None
        } else {
            let val = self.parse_expression(Precedence::Lowest)?;
            end = val.span;
            Some(val)
        };

        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement {
            kind: StatementKind::Return(return_value),
            span: start.join(end),
        })
    }
    
    fn parse_print_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        if !self.expect_peek(Token::LParen) {
            return None;
        }
        
        let mut expressions = Vec::new();
        if self.peek_token_is(&Token::RParen) {
            self.next_token();
        } else {
            self.next_token();
            expressions.push(self.parse_expression(Precedence::Lowest)?);
            
            while self.peek_token_is(&Token::Comma) {
                self.next_token();
                self.next_token();
                expressions.push(self.parse_expression(Precedence::Lowest)?);
            }
            
            if !self.expect_peek(Token::RParen) {
                return None;
            }
        }
        
        let end = self.cur_span;
        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }
        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }
        
        Some(Statement {
            kind: StatementKind::Print(expressions),
            span: start.join(end),
        })
    }

    fn parse_import_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip import

        let path = match &self.cur_token {
            Token::String(s) => s.clone(),
            Token::Identifier(s) => s.clone(),
            _ => {
                let msg = format!("Expected string or identifier after import, got {:?}", self.cur_token);
                self.errors.push(ParserError { message: msg, span: self.cur_span });
                return None;
            }
        };

        let mut alias = None;
        let mut end = self.cur_span;

        if self.peek_token_is_identifier("as") {
            self.next_token(); // as
            if !self.expect_peek(Token::Identifier("".to_string())) {
                return None;
            }
            alias = match &self.cur_token {
                Token::Identifier(s) => Some(s.clone()),
                _ => None,
            };
            end = self.cur_span;
        }

        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement {
            kind: StatementKind::Import { path, alias },
            span: start.join(end),
        })
    }

    fn parse_expression_statement(&mut self) -> Option<Statement> {
        let expr = self.parse_expression(Precedence::Lowest)?;
        let start = expr.span;

        if self.peek_token_is(&Token::Assign) 
            || self.peek_token_is(&Token::PlusAssign)
            || self.peek_token_is(&Token::MinusAssign)
            || self.peek_token_is(&Token::StarAssign)
            || self.peek_token_is(&Token::SlashAssign)
            || self.peek_token_is(&Token::PercentAssign)
            || self.peek_token_is(&Token::DoubleStarAssign)
            || self.peek_token_is(&Token::AmpersandAssign)
            || self.peek_token_is(&Token::PipeAssign)
            || self.peek_token_is(&Token::CaretAssign)
            || self.peek_token_is(&Token::ShiftLeftAssign)
            || self.peek_token_is(&Token::ShiftRightAssign) {
            
            let tok = self.peek_token.clone();
            self.next_token(); // consume expr
            self.next_token(); // consume operator
            let value = self.parse_expression(Precedence::Lowest)?;
            
            let op = match tok {
                Token::PlusAssign => Some(InfixOperator::Plus),
                Token::MinusAssign => Some(InfixOperator::Minus),
                Token::StarAssign => Some(InfixOperator::Multiply),
                Token::SlashAssign => Some(InfixOperator::Divide),
                Token::PercentAssign => Some(InfixOperator::Modulo),
                Token::DoubleStarAssign => Some(InfixOperator::Power),
                Token::AmpersandAssign => Some(InfixOperator::BitwiseAnd),
                Token::PipeAssign => Some(InfixOperator::BitwiseOr),
                Token::CaretAssign => Some(InfixOperator::BitwiseXor),
                Token::ShiftLeftAssign => Some(InfixOperator::ShiftLeft),
                Token::ShiftRightAssign => Some(InfixOperator::ShiftRight),
                _ => None,
            };

            match expr.kind {
                ExpressionKind::Index { object, index } => {
                    let final_value = if let Some(operator) = op {
                        Expression {
                            kind: ExpressionKind::Infix {
                                left: Box::new(Expression { kind: ExpressionKind::Index { object: object.clone(), index: index.clone() }, span: expr.span }),
                                operator,
                                right: Box::new(value.clone()),
                            },
                            span: expr.span.join(value.span)
                        }
                    } else {
                        value.clone()
                    };
                    return Some(Statement {
                        span: start.join(value.span),
                        kind: StatementKind::IndexAssign {
                            object: *object,
                            index: *index,
                            value: final_value,
                        },
                    });
                }
                ExpressionKind::Identifier(name) => {
                    let final_value = if let Some(operator) = op {
                        Expression {
                            kind: ExpressionKind::Infix {
                                left: Box::new(Expression { kind: ExpressionKind::Identifier(name.clone()), span: expr.span }),
                                operator,
                                right: Box::new(value.clone()),
                            },
                            span: expr.span.join(value.span)
                        }
                    } else {
                        value.clone()
                    };
                    return Some(Statement {
                        span: start.join(value.span),
                        kind: StatementKind::Let { names: vec![name], value: final_value, type_hint: None },
                    });
                }
                _ => {
                    self.errors.push(ParserError { message: format!("Invalid assignment target"), span: start });
                    return None;
                }
            }
        }

        if self.peek_token_is(&Token::Semicolon) {
            self.next_token();
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement {
            span: start,
            kind: StatementKind::Expression(expr),
        })
    }

    fn parse_block(&mut self) -> Block {
        let start = self.cur_span;
        if !self.cur_token_is(&Token::Indent) {
            while self.cur_token_is(&Token::Newline) {
                self.next_token();
            }
            if !self.cur_token_is(&Token::Indent) {
                self.errors.push(ParserError { message: format!("Expected Indent"), span: self.cur_span });
                return Block { statements: vec![], span: self.cur_span };
            }
        }
        
        self.next_token(); // consume Indent

        let mut statements = Vec::new();
        while !self.cur_token_is(&Token::Dedent) && !self.cur_token_is(&Token::EOF) {
            if let Some(stmt) = self.parse_statement() {
                statements.push(stmt);
            }
            self.next_token();
            
            while self.cur_token_is(&Token::Newline) {
                self.next_token();
            }
        }

        let end = self.cur_span;
        Block { statements, span: start.join(end) }
    }

    fn parse_function_def(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        if !self.expect_peek(Token::Identifier("".to_string())) {
            return None;
        }
        
        let name = match &self.cur_token {
            Token::Identifier(n) => n.clone(),
            _ => return None,
        };

        if !self.expect_peek(Token::LParen) {
            return None;
        }

        let params = self.parse_function_params()?;
        
        let mut return_type = None;
        if self.peek_token_is(&Token::Arrow) {
            self.next_token(); // )
            self.next_token(); // ->
            return_type = self.parse_type();
        }

        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        if !self.expect_peek(Token::Newline) {
            return None;
        }
        self.next_token(); // consume Newline
        
        let body = self.parse_block();
        let end = body.span;
        
        Some(Statement {
            span: start.join(end),
            kind: StatementKind::FunctionDef { name, params, body, return_type }
        })
    }

    fn parse_function_params(&mut self) -> Option<Vec<(String, Option<crate::ast::Type>)>> {
        let mut params = Vec::new();

        if self.peek_token_is(&Token::RParen) {
            self.next_token();
            return Some(params);
        }

        self.next_token();

        let name = match &self.cur_token {
            Token::Identifier(ident) => ident.clone(),
            _ => return None,
        };
        
        let mut p_type = None;
        if self.peek_token_is(&Token::Colon) {
            self.next_token(); // identifier
            self.next_token(); // :
            p_type = self.parse_type();
        }
        params.push((name, p_type));

        while self.peek_token_is(&Token::Comma) {
            self.next_token(); // comma
            self.next_token(); // next identifier
            let name = match &self.cur_token {
                Token::Identifier(ident) => ident.clone(),
                _ => return None,
            };
            let mut p_type = None;
            if self.peek_token_is(&Token::Colon) {
                self.next_token(); // identifier
                self.next_token(); // :
                p_type = self.parse_type();
            }
            params.push((name, p_type));
        }

        if !self.expect_peek(Token::RParen) {
            return None;
        }

        Some(params)
    }

    fn parse_if_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip if
        let condition = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        if !self.expect_peek(Token::Newline) {
            return None;
        }
        self.next_token(); // consume Newline

        let consequence = self.parse_block();
        let mut elif_branches = Vec::new();
        let mut alternative = None;
        let mut end = consequence.span;

        while self.cur_token_is(&Token::Dedent) && self.peek_token_is(&Token::Elif) {
            self.next_token(); // move to elif
            self.next_token(); // skip elif
            let elif_condition = self.parse_expression(Precedence::Lowest)?;
            if !self.expect_peek(Token::Colon) {
                return None;
            }
            if !self.expect_peek(Token::Newline) {
                return None;
            }
            self.next_token(); // consume Newline
            let elif_consequence = self.parse_block();
            end = elif_consequence.span;
            elif_branches.push((elif_condition, elif_consequence));
        }

        if self.cur_token_is(&Token::Dedent) && self.peek_token_is(&Token::Else) {
            self.next_token(); // move to else
            self.next_token(); // move to :
            
            if !self.cur_token_is(&Token::Colon) {
                 self.errors.push(ParserError { message: format!("Expected Colon after else"), span: self.cur_span });
                 return None;
            }
            
            if !self.expect_peek(Token::Newline) {
                return None;
            }
            self.next_token(); // consume Newline
            let alt_block = self.parse_block();
            end = alt_block.span;
            alternative = Some(alt_block);
        }

        Some(Statement {
            span: start.join(end),
            kind: StatementKind::If { condition, consequence, elif_branches, alternative }
        })
    }
    
    fn parse_while_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip while
        let condition = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        if !self.expect_peek(Token::Newline) {
            return None;
        }

        let body = self.parse_block();
        let end = body.span;
        
        Some(Statement {
            span: start.join(end),
            kind: StatementKind::While { condition, body }
        })
    }

    fn parse_for_statement(&mut self) -> Option<Statement> {
        let start = self.cur_span;
        self.next_token(); // skip for

        if !self.cur_token_is(&Token::Identifier("".to_string())) {
            self.errors.push(ParserError { message: format!("Expected Identifier after 'for'"), span: self.cur_span });
            return None;
        }

        let variable = match &self.cur_token {
            Token::Identifier(n) => n.clone(),
            _ => return None,
        };

        if !self.expect_peek(Token::In) {
            return None;
        }

        self.next_token(); // skip in
        let iterable = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(Token::Colon) {
            return None;
        }

        if !self.expect_peek(Token::Newline) {
            return None;
        }
        self.next_token(); // consume Newline

        let body = self.parse_block();
        let end = body.span;

        Some(Statement {
            span: start.join(end),
            kind: StatementKind::For { variable, iterable, body }
        })
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        let start = self.cur_span;
        let mut left_expr = match &self.cur_token {
            Token::Identifier(i) => Some(Expression { kind: ExpressionKind::Identifier(i.clone()), span: start }),
            Token::Integer(i) => Some(Expression { kind: ExpressionKind::Integer(*i), span: start }),
            Token::Float(f) => Some(Expression { kind: ExpressionKind::Float(*f), span: start }),
            Token::String(s) => Some(Expression { kind: ExpressionKind::String(s.clone()), span: start }),
            Token::Minus | Token::Not | Token::Tilde => {
                self.parse_prefix_expression()
            },
            Token::LParen => {
                self.next_token();
                let expr = self.parse_expression(Precedence::Lowest);
                if !self.expect_peek(Token::RParen) {
                    return None;
                }
                expr
            },
            Token::LBracket => self.parse_list_literal(),
            Token::LBrace => self.parse_brace_literal(),
            Token::Lambda => self.parse_lambda_expression(),
            Token::FString(_) => self.parse_fstring(),
            _ => {
                let msg = format!("Expected expression, got {:?}", self.cur_token);
                self.errors.push(ParserError { message: msg, span: self.cur_span });
                None
            },
        }?;

        while !self.peek_token_is(&Token::Newline) && !self.peek_token_is(&Token::EOF) && precedence < self.peek_precedence() {
            match self.peek_token {
                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent | Token::Equal | Token::NotEqual | Token::LessThan | Token::GreaterThan | Token::LessThanOrEqual | Token::GreaterThanOrEqual | Token::In | Token::NotIn | Token::At | Token::DoubleStar | Token::And | Token::Or | Token::Ampersand | Token::Pipe | Token::Caret | Token::ShiftLeft | Token::ShiftRight => {
                    self.next_token();
                    left_expr = self.parse_infix_expression(left_expr)?;
                },
                Token::LParen => {
                    self.next_token();
                    left_expr = self.parse_call_expression(left_expr)?;
                },
                Token::Dot => {
                    self.next_token();
                    left_expr = self.parse_get_expression(left_expr)?;
                }
                Token::LBracket => {
                self.next_token(); // cur_token = [
                self.next_token(); // cur_token = start of index
                left_expr = self.parse_index_expression(left_expr)?
            }
            Token::If => {
                self.next_token();
                left_expr = self.parse_ternary_expression(left_expr)?
            }
            _ => break,
            }
        }

        Some(left_expr)
    }

    fn parse_list_literal(&mut self) -> Option<Expression> {
        let start = self.cur_span;
        let mut elements = Vec::new();

        if self.peek_token_is(&Token::RBracket) {
            self.next_token(); // move to ]
            return Some(Expression { kind: ExpressionKind::List(elements), span: start.join(self.cur_span) });
        }

        self.next_token(); // skip [
        elements.push(self.parse_expression(Precedence::Lowest)?);

        if self.peek_token_is(&Token::For) {
            self.next_token(); // skip for
            if !self.expect_peek(Token::Identifier("".to_string())) {
                return None;
            }
            let variable = match &self.cur_token {
                Token::Identifier(n) => n.clone(),
                _ => return None,
            };
            if !self.expect_peek(Token::In) {
                return None;
            }
            self.next_token(); // skip in
            let iterable = self.parse_expression(Precedence::Ternary)?;
            
            let mut condition = None;
            if self.peek_token_is(&Token::If) {
                self.next_token();
                self.next_token();
                condition = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
            }

            if !self.expect_peek(Token::RBracket) {
                return None;
            }
            let end = self.cur_span;
            return Some(Expression {
                kind: ExpressionKind::ListComprehension {
                    element: Box::new(elements.remove(0)),
                    variable,
                    iterable: Box::new(iterable),
                    condition,
                },
                span: start.join(end),
            });
        }

        while self.peek_token_is(&Token::Comma) {
            self.next_token();
            self.next_token();
            elements.push(self.parse_expression(Precedence::Lowest)?);
        }

        if !self.expect_peek(Token::RBracket) {
            return None;
        }

        let end = self.cur_span;
        Some(Expression { kind: ExpressionKind::List(elements), span: start.join(end) })
    }

    fn parse_brace_literal(&mut self) -> Option<Expression> {
        let start = self.cur_span;
        self.next_token(); // skip {
        
        if self.cur_token_is(&Token::RBrace) {
            return Some(Expression { kind: ExpressionKind::Dictionary(vec![]), span: start.join(self.cur_span) });
        }
        
        // Parse first expression
        let first_expr = self.parse_expression(Precedence::Lowest)?;
        
        if self.peek_token_is(&Token::Colon) {
            // Dictionary or Dict Comprehension
            self.next_token(); // move to first_expr
            self.next_token(); // move to :
            let first_val = self.parse_expression(Precedence::Lowest)?;
            
            if self.peek_token_is(&Token::For) {
                // {k: v for x in y}
                self.next_token(); // move to for
                if !self.expect_peek(Token::Identifier("".to_string())) { return None; }
                let variable = match &self.cur_token {
                    Token::Identifier(n) => n.clone(),
                    _ => return None,
                };
                if !self.expect_peek(Token::In) { return None; }
                self.next_token(); // skip in
                let iterable = self.parse_expression(Precedence::Ternary)?;
                
                let mut condition = None;
                if self.peek_token_is(&Token::If) {
                    self.next_token(); // skip to if
                    self.next_token(); // skip if
                    condition = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
                }
                
                if !self.expect_peek(Token::RBrace) { return None; }
                let end = self.cur_span;
                return Some(Expression { 
                    kind: ExpressionKind::DictComprehension { 
                        key: Box::new(first_expr), 
                        value: Box::new(first_val), 
                        variable, 
                        iterable: Box::new(iterable), 
                        condition 
                    }, 
                    span: start.join(end) 
                });
            } else {
                // Dictionary literal
                let mut pairs = vec![(first_expr, first_val)];
                while self.peek_token_is(&Token::Comma) {
                    self.next_token(); // skip ,
                    self.next_token(); // move to next key
                    let key = self.parse_expression(Precedence::Lowest)?;
                    if !self.expect_peek(Token::Colon) { return None; }
                    self.next_token(); // skip :
                    let val = self.parse_expression(Precedence::Lowest)?;
                    pairs.push((key, val));
                }
                
                if !self.expect_peek(Token::RBrace) { return None; }
                let end = self.cur_span;
                Some(Expression { kind: ExpressionKind::Dictionary(pairs), span: start.join(end) })
            }
        } else {
            // Set or Set Comprehension
            if self.peek_token_is(&Token::For) {
                // {x for x in y}
                self.next_token(); // move to for
                if !self.expect_peek(Token::Identifier("".to_string())) { return None; }
                let variable = match &self.cur_token {
                    Token::Identifier(n) => n.clone(),
                    _ => return None,
                };
                if !self.expect_peek(Token::In) { return None; }
                self.next_token(); // skip in
                let iterable = self.parse_expression(Precedence::Ternary)?;
                
                let mut condition = None;
                if self.peek_token_is(&Token::If) {
                    self.next_token(); // skip to if
                    self.next_token(); // skip if
                    condition = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
                }
                
                if !self.expect_peek(Token::RBrace) { return None; }
                let end = self.cur_span;
                return Some(Expression { 
                    kind: ExpressionKind::SetComprehension { 
                        element: Box::new(first_expr), 
                        variable, 
                        iterable: Box::new(iterable), 
                        condition 
                    }, 
                    span: start.join(end) 
                });
            } else {
                // Set literal
                let mut elements = vec![first_expr];
                while self.peek_token_is(&Token::Comma) {
                    self.next_token(); // skip ,
                    self.next_token(); // move to next element
                    elements.push(self.parse_expression(Precedence::Lowest)?);
                }
                
                if !self.expect_peek(Token::RBrace) { return None; }
                let end = self.cur_span;
                Some(Expression { kind: ExpressionKind::Set(elements), span: start.join(end) })
            }
        }
    }
    
    fn parse_lambda_expression(&mut self) -> Option<Expression> {
        let start = self.cur_span;
        self.next_token(); // skip lambda
        
        let mut params = Vec::new();
        if !self.cur_token_is(&Token::Colon) {
            if let Token::Identifier(name) = &self.cur_token {
                params.push(name.clone());
            } else {
                self.errors.push(ParserError { message: format!("Expected Identifier for lambda param, got {:?}", self.cur_token), span: self.cur_span });
                return None;
            }
            
            while self.peek_token_is(&Token::Comma) {
                self.next_token(); // comma
                if !self.expect_peek(Token::Identifier("".to_string())) {
                    return None;
                }
                if let Token::Identifier(name) = &self.cur_token {
                    params.push(name.clone());
                }
            }
        }
        
        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        self.next_token(); // skip :
        let body = self.parse_expression(Precedence::Lowest)?;
        
        Some(Expression { 
            span: start.join(body.span),
            kind: ExpressionKind::Lambda { params, body: Box::new(body) }
        })
    }

    fn parse_fstring(&mut self) -> Option<Expression> {
        let content = if let Token::FString(s) = &self.cur_token {
            s.clone()
        } else {
            return None;
        };
        let start_span = self.cur_span;
        
        let mut parts = Vec::new();
        let mut current_literal = String::new();
        let chars_vec: Vec<char> = content.chars().collect();
        let mut i = 0;
        
        while i < chars_vec.len() {
            let ch = chars_vec[i];
            if ch == '{' {
                // If double {{, it's an escaped {
                if i + 1 < chars_vec.len() && chars_vec[i+1] == '{' {
                    current_literal.push('{');
                    i += 2;
                    continue;
                }
                
                // Finish current literal
                if !current_literal.is_empty() {
                    parts.push(crate::ast::FStringPart::Literal(current_literal.clone()));
                    current_literal.clear();
                }
                
                // Extract expression content
                let mut expr_text = String::new();
                let mut brace_level = 1;
                i += 1;
                while i < chars_vec.len() && brace_level > 0 {
                    let inner_ch = chars_vec[i];
                    if inner_ch == '{' { brace_level += 1; }
                    else if inner_ch == '}' { brace_level -= 1; }
                    
                    if brace_level > 0 {
                        expr_text.push(inner_ch);
                    }
                    i += 1;
                }
                
                if brace_level > 0 {
                    self.errors.push(ParserError { message: "Unclosed { in f-string".to_string(), span: start_span });
                    return None;
                }
                
                // Parse expression text
                let lexer = Lexer::new(&expr_text);
                let mut parser = Parser::new(lexer);
                if let Some(expr) = parser.parse_expression(Precedence::Lowest) {
                    parts.push(crate::ast::FStringPart::Expression(Box::new(expr)));
                } else {
                    self.errors.push(ParserError { message: format!("Failed to parse expression in f-string: {}", expr_text), span: start_span });
                    return None;
                }
            } else if ch == '}' {
                if i + 1 < chars_vec.len() && chars_vec[i+1] == '}' {
                    current_literal.push('}');
                    i += 2;
                } else {
                    self.errors.push(ParserError { message: "Single } found in f-string".to_string(), span: start_span });
                    return None;
                }
            } else {
                current_literal.push(ch);
                i += 1;
            }
        }
        
        if !current_literal.is_empty() {
            parts.push(crate::ast::FStringPart::Literal(current_literal));
        }

        Some(Expression { 
            span: start_span,
            kind: ExpressionKind::FString(parts)
        })
    }

    fn parse_index_expression(&mut self, left: Expression) -> Option<Expression> {
        let start_span = left.span;
        
        let mut start_index = None;
        let mut is_slice = false;

        if !self.cur_token_is(&Token::Colon) {
            start_index = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
        } else {
             is_slice = true;
        }

        if self.peek_token_is(&Token::Colon) || (is_slice && self.cur_token_is(&Token::Colon)) {
            // It's a slice
            if !is_slice {
                 self.next_token(); // consume start index, now at :
            }
            
            // Should be at Colon now.
            if !self.cur_token_is(&Token::Colon) {
                 // Logic error or unexpected state
                 self.errors.push(ParserError { message: format!("Expected : in slice"), span: self.cur_span });
                 return None;
            }
        } else {
            // Normal index
            if !self.expect_peek(Token::RBracket) {
                return None;
            }
            let end_span = self.cur_span;
            return Some(Expression {
                kind: ExpressionKind::Index {
                    object: Box::new(left),
                    index: start_index.unwrap(),
                },
                span: start_span.join(end_span),
            });
        }
        
        return self.parse_slice_inner(left, start_index);
    }

    fn parse_slice_inner(&mut self, left: Expression, start_expr: Option<Box<Expression>>) -> Option<Expression> {
        let start_span = left.span;
        
        // We are at the first Colon
        if !self.cur_token_is(&Token::Colon) {
             return None;
        }
        
        self.next_token(); // move past Colon
        
        let mut end_expr = None;
        if !self.cur_token_is(&Token::Colon) && !self.cur_token_is(&Token::RBracket) {
            end_expr = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
             // After parse_expression, cur_token is the last token of expr. 
             // We need to peek to decide if we continue.
             // But Wait. parse_expression consumes tokens.
             // If we have 1:2, parse_expression(1) stops at 1. peek is :.
             // parse_index_expression logic:
             // start_index parsed. peek is :. next_token() -> cur is :.
             // parse_slice_inner called. cur is :.
             // next_token() -> cur is 2 (or whatever).
             // parse_expression -> parses 2. cur is 2. peek is ] or :.
             
             // So if we parsed end_expr, we need to advance if the next token is : or ].
             // But parse_expression doesn't advance past the expression? 
             // Yes it does, it consumes the expression.
        }
        
        if self.peek_token_is(&Token::Colon) {
            self.next_token(); // move to :
        } else if self.peek_token_is(&Token::RBracket) {
            self.next_token(); // move to ]
        }
        
        let mut step_expr = None;
        if self.cur_token_is(&Token::Colon) {
            self.next_token(); // skip :
            if !self.cur_token_is(&Token::RBracket) {
                 step_expr = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
                 if self.peek_token_is(&Token::RBracket) {
                     self.next_token();
                 }
            }
        }
        
        if !self.cur_token_is(&Token::RBracket) {
             self.errors.push(ParserError { message: format!("Expected ] after slice, got {:?}", self.cur_token), span: self.cur_span });
             return None;
        }
        
        let slice_expr = Expression {
            kind: ExpressionKind::Slice {
                 start: start_expr,
                 end: end_expr,
                 step: step_expr
            },
            span: start_span.join(self.cur_span)
        };
        
        Some(Expression {
            kind: ExpressionKind::Index {
                object: Box::new(left),
                index: Box::new(slice_expr),
            },
            span: start_span.join(self.cur_span)
        })
    }

    fn parse_prefix_expression(&mut self) -> Option<Expression> {
        let start = self.cur_span;
        let operator = match self.cur_token {
            Token::Minus => PrefixOperator::Minus,
            Token::Not => PrefixOperator::Not,
            Token::Tilde => PrefixOperator::BitwiseNot,
            _ => return None,
        };
        
        self.next_token();
        let right = self.parse_expression(Precedence::Prefix)?;
        let end = right.span;
        
        Some(Expression {
            kind: ExpressionKind::Prefix { operator, right: Box::new(right) },
            span: start.join(end),
        })
    }

    fn parse_infix_expression(&mut self, left: Expression) -> Option<Expression> {
        let start = left.span;
        let operator = self.cur_token.to_infix_operator()?;
        let precedence = self.cur_precedence();
        self.next_token();
        let right = self.parse_expression(precedence)?;
        let end = right.span;
        Some(Expression {
            kind: ExpressionKind::Infix {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            },
            span: start.join(end),
        })
    }

    fn parse_ternary_expression(&mut self, consequence: Expression) -> Option<Expression> {
        let start = consequence.span;
        self.next_token(); // move past if
        
        // condition
        let condition = self.parse_expression(Precedence::Ternary)?;
        
        if !self.expect_peek(Token::Else) {
            return None;
        }
        self.next_token(); // move past else
        
        // alternative
        let alternative = self.parse_expression(Precedence::Ternary)?;
        
        let end = alternative.span;
        Some(Expression {
            kind: ExpressionKind::Ternary {
                condition: Box::new(condition),
                consequence: Box::new(consequence),
                alternative: Box::new(alternative),
            },
            span: start.join(end),
        })
    }
    
    fn parse_call_expression(&mut self, function: Expression) -> Option<Expression> {
        let start = function.span;
        let arguments = self.parse_call_arguments()?;
        let end = self.cur_span;
        Some(Expression {
            kind: ExpressionKind::Call { function: Box::new(function), arguments },
            span: start.join(end),
        })
    }
    
    fn parse_call_arguments(&mut self) -> Option<Vec<Expression>> {
        let mut args = Vec::new();

        if self.peek_token_is(&Token::RParen) {
            self.next_token();
            return Some(args);
        }

        self.next_token();
        args.push(self.parse_expression(Precedence::Lowest)?);

        while self.peek_token_is(&Token::Comma) {
            self.next_token();
            self.next_token();
            args.push(self.parse_expression(Precedence::Lowest)?);
        }

        if !self.expect_peek(Token::RParen) {
            return None;
        }

        Some(args)
    }

    fn parse_get_expression(&mut self, object: Expression) -> Option<Expression> {
        let start = object.span;
        
        let method_name = match &self.peek_token {
            Token::Identifier(name) => name.clone(),
            Token::Print => "print".to_string(),
            Token::Def => "def".to_string(),
            Token::Return => "return".to_string(),
            Token::If => "if".to_string(),
            Token::Else => "else".to_string(),
            Token::Elif => "elif".to_string(),
            Token::While => "while".to_string(),
            Token::Let => "let".to_string(),
            Token::And => "and".to_string(),
            Token::Or => "or".to_string(),
            Token::For => "for".to_string(),
            Token::In => "in".to_string(),
            Token::NotIn => "not in".to_string(),
            Token::Not => "not".to_string(),
            Token::Break => "break".to_string(),
            Token::Continue => "continue".to_string(),
            Token::Import => "import".to_string(),
            Token::Assert => "assert".to_string(),
            _ => {
                self.errors.push(ParserError { message: format!("Expected Identifier or Keyword after '.'"), span: self.peek_span });
                return None;
            }
        };

        self.next_token();
        let end = self.cur_span;
        
        // Check if this is a method call (obj.method(...))
        if self.peek_token_is(&Token::LParen) {
            self.next_token(); // move to (
            let arguments = self.parse_call_arguments()?;
            let end_span = self.cur_span;
            Some(Expression {
                kind: ExpressionKind::MethodCall { 
                    object: Box::new(object), 
                    method: method_name, 
                    arguments 
                },
                span: start.join(end_span),
            })
        } else {
            // Regular property access
            Some(Expression {
                kind: ExpressionKind::Get { object: Box::new(object), name: method_name },
                span: start.join(end),
            })
        }
    }

    fn peek_precedence(&self) -> Precedence {
        match self.peek_token {
            Token::If => Precedence::Ternary,
            Token::Or => Precedence::LogicalOr,
            Token::And => Precedence::LogicalAnd,
            Token::Equal | Token::NotEqual | Token::LessThan | Token::GreaterThan | Token::LessThanOrEqual | Token::GreaterThanOrEqual | Token::In | Token::NotIn => Precedence::Equals,
            Token::Plus | Token::Minus => Precedence::Sum,
            Token::ShiftLeft | Token::ShiftRight => Precedence::Shift,
            Token::Ampersand => Precedence::BitwiseAnd,
            Token::Caret => Precedence::BitwiseXor,
            Token::Pipe => Precedence::BitwiseOr,
            Token::Star | Token::Slash | Token::Percent | Token::At => Precedence::Product,
            Token::DoubleStar => Precedence::Power,
            Token::LParen => Precedence::Call,
            Token::Dot | Token::LBracket => Precedence::Index,
            _ => Precedence::Lowest,
        }
    }

    fn cur_precedence(&self) -> Precedence {
        match self.cur_token {
            Token::If => Precedence::Ternary,
            Token::Or => Precedence::LogicalOr,
            Token::And => Precedence::LogicalAnd,
            Token::Equal | Token::NotEqual | Token::LessThan | Token::GreaterThan | Token::LessThanOrEqual | Token::GreaterThanOrEqual => Precedence::Equals,
            Token::Plus | Token::Minus => Precedence::Sum,
            Token::ShiftLeft | Token::ShiftRight => Precedence::Shift,
            Token::Ampersand => Precedence::BitwiseAnd,
            Token::Caret => Precedence::BitwiseXor,
            Token::Pipe => Precedence::BitwiseOr,
            Token::Star | Token::Slash | Token::Percent | Token::At => Precedence::Product,
            Token::DoubleStar => Precedence::Power,
            Token::LParen => Precedence::Call,
            Token::Dot | Token::LBracket => Precedence::Index,
            _ => Precedence::Lowest,
        }
    }

    fn expect_peek(&mut self, t: Token) -> bool {
        match (&self.peek_token, &t) {
            (Token::Identifier(_), Token::Identifier(_)) => {
                self.next_token();
                true
            },
            (t1, t2) if t1 == t2 => {
                self.next_token();
                true
            },
            _ => {
                self.peek_error(t);
                false
            }
        }
    }

    fn peek_error(&mut self, t: Token) {
        let msg = format!("Expected {:?}, got {:?}", t, self.peek_token);
        self.errors.push(ParserError { message: msg, span: self.peek_span });
    }

    fn peek_token_is(&self, t: &Token) -> bool {
        match (&self.peek_token, t) {
            (Token::Identifier(_), Token::Identifier(_)) => true,
            (t1, t2) => t1 == t2,
        }
    }
    
    fn cur_token_is(&self, t: &Token) -> bool {
        match (&self.cur_token, t) {
            (Token::Identifier(_), Token::Identifier(_)) => true,
            (t1, t2) => t1 == t2,
        }
    }

    fn peek_token_is_identifier(&self, name: &str) -> bool {
        match &self.peek_token {
            Token::Identifier(n) => n == name,
            _ => false,
        }
    }
}
