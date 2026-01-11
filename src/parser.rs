use crate::token::Token;
use crate::lexer::Lexer;
use crate::ast::{Statement, Expression, Block, InfixOperator, PrefixOperator};

#[derive(PartialEq, PartialOrd)]
enum Precedence {
    Lowest,
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
    peek_token: Token,
    pub errors: Vec<String>,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut p = Parser {
            lexer,
            cur_token: Token::EOF,
            peek_token: Token::EOF,
            errors: Vec::new(),
        };
        // Read two tokens to initialize cur and peek
        p.next_token();
        p.next_token();
        p
    }

    fn next_token(&mut self) {
        self.cur_token = self.peek_token.clone();
        self.peek_token = self.lexer.next_token();
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
            Token::Break => {
                self.next_token();
                if self.peek_token_is(&Token::Newline) { self.next_token(); }
                Some(Statement::Break)
            }
            Token::Continue => {
                self.next_token();
                if self.peek_token_is(&Token::Newline) { self.next_token(); }
                Some(Statement::Continue)
            }
            Token::Newline => None, // Skip empty lines
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Option<Statement> {
        // let x = 5
        // Or Python style: x = 5 (which is expression statement assignment?)
        // User asked for Python simplicity but Rust safety. "let" is explicit.
        // Let's support "let x = 5" for now as per plan.
        
        // Expect Identifier
        if !self.expect_peek(Token::Identifier("".to_string())) {
            return None;
        }
        
        let name = match &self.cur_token {
            Token::Identifier(n) => n.clone(),
            _ => return None,
        };

        if !self.expect_peek(Token::Assign) {
            return None;
        }

        self.next_token(); // skip =

        let value = self.parse_expression(Precedence::Lowest)?;

        // Optional newline/semicolon? Python uses newline.
        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement::Let { name, value })
    }

    fn parse_return_statement(&mut self) -> Option<Statement> {
        self.next_token(); // skip return

        let return_value = if self.cur_token_is(&Token::Newline) || self.cur_token_is(&Token::EOF) {
            None
        } else {
            Some(self.parse_expression(Precedence::Lowest)?)
        };

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement::Return(return_value))
    }
    
    fn parse_print_statement(&mut self) -> Option<Statement> {
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
        
        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }
        
        Some(Statement::Print(expressions))
    }

    fn parse_expression_statement(&mut self) -> Option<Statement> {
        let expr = self.parse_expression(Precedence::Lowest)?;

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

            match expr {
                Expression::Index { object, index } => {
                    let final_value = if let Some(operator) = op {
                        Expression::Infix {
                            left: Box::new(Expression::Index { object: object.clone(), index: index.clone() }),
                            operator,
                            right: Box::new(value),
                        }
                    } else {
                        value
                    };
                    return Some(Statement::IndexAssign {
                        object: *object,
                        index: *index,
                        value: final_value,
                    });
                }
                Expression::Identifier(name) => {
                    let final_value = if let Some(operator) = op {
                        Expression::Infix {
                            left: Box::new(Expression::Identifier(name.clone())),
                            operator,
                            right: Box::new(value),
                        }
                    } else {
                        value
                    };
                    return Some(Statement::Let { name, value: final_value });
                }
                _ => {
                    self.errors.push(format!("Invalid assignment target: {:?}", expr));
                    return None;
                }
            }
        }

        if self.peek_token_is(&Token::Newline) {
            self.next_token();
        }

        Some(Statement::Expression(expr))
    }

    fn parse_block(&mut self) -> Block {
        // Expect Indent
        if !self.cur_token_is(&Token::Indent) {
            // Error? Or maybe single line block?
            // For MVP, enforce Indent.
            // But wait, caller consumes Colon and Newline usually.
            // So cur_token should be Indent.
            // If not, maybe we are at Newline?
            while self.cur_token_is(&Token::Newline) {
                self.next_token();
            }
            if !self.cur_token_is(&Token::Indent) {
                self.errors.push(format!("Expected Indent, got {:?}", self.cur_token));
                return Block { statements: vec![] };
            }
        }
        
        self.next_token(); // consume Indent

        let mut statements = Vec::new();

        while !self.cur_token_is(&Token::Dedent) && !self.cur_token_is(&Token::EOF) {
            if let Some(stmt) = self.parse_statement() {
                statements.push(stmt);
            }
            self.next_token();
            
            // Skip extra newlines
            while self.cur_token_is(&Token::Newline) {
                self.next_token();
            }
        }

        // consume Dedent (handled by caller or here?)
        // If loop ended on Dedent, we are ON Dedent.
        // Caller might expect us to consume it?
        // Let's consume it here.
        if self.cur_token_is(&Token::Dedent) {
            // self.next_token(); // Do NOT consume here if we want symmetry? 
            // Usually parse_block consumes the block content.
            // Let's consume Dedent.
        }
        
        Block { statements }
    }

    fn parse_function_def(&mut self) -> Option<Statement> {
        // def name(params):
        //    block
        
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

        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        if !self.expect_peek(Token::Newline) {
            return None;
        }
        self.next_token(); // consume Newline
        
        // Now we should be at Indent (or Newline then Indent)
        let body = self.parse_block();
        
        Some(Statement::FunctionDef { name, params, body })
    }

    fn parse_function_params(&mut self) -> Option<Vec<String>> {
        let mut params = Vec::new();

        if self.peek_token_is(&Token::RParen) {
            self.next_token();
            return Some(params);
        }

        self.next_token();

        match &self.cur_token {
            Token::Identifier(ident) => params.push(ident.clone()),
            _ => return None,
        }

        while self.peek_token_is(&Token::Comma) {
            self.next_token();
            self.next_token();
            match &self.cur_token {
                Token::Identifier(ident) => params.push(ident.clone()),
                _ => return None,
            }
        }

        if !self.expect_peek(Token::RParen) {
            return None;
        }

        Some(params)
    }

    fn parse_if_statement(&mut self) -> Option<Statement> {
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

        // Check for elif
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
            elif_branches.push((elif_condition, elif_consequence));
        }

        // Check for else
        // parse_block ends at Dedent.
        if self.cur_token_is(&Token::Dedent) && self.peek_token_is(&Token::Else) {
            self.next_token(); // move to else
            self.next_token(); // move to :
            
            if !self.cur_token_is(&Token::Colon) {
                 self.errors.push(format!("Expected Colon after else, got {:?}", self.cur_token));
                 return None;
            }
            
            if !self.expect_peek(Token::Newline) {
                return None;
            }
            self.next_token(); // consume Newline
            alternative = Some(self.parse_block());
        }

        Some(Statement::If { condition, consequence, elif_branches, alternative })
    }
    
    fn parse_while_statement(&mut self) -> Option<Statement> {
        self.next_token(); // skip while
        let condition = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(Token::Colon) {
            return None;
        }
        
        if !self.expect_peek(Token::Newline) {
            return None;
        }

        let body = self.parse_block();
        
        Some(Statement::While { condition, body })
    }

    fn parse_for_statement(&mut self) -> Option<Statement> {
        self.next_token(); // skip for

        // Expect Identifier
        if !self.cur_token_is(&Token::Identifier("".to_string())) {
            self.errors.push(format!("Expected Identifier after 'for', got {:?}", self.cur_token));
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

        Some(Statement::For { variable, iterable, body })
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        let mut left_expr = match &self.cur_token {
            Token::Identifier(i) => Some(Expression::Identifier(i.clone())),
            Token::Integer(i) => Some(Expression::Integer(*i)),
            Token::Float(f) => Some(Expression::Float(*f)),
            Token::String(s) => Some(Expression::String(s.clone())),
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
            Token::LBrace => self.parse_dictionary_literal(),
            _ => None,
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
                    left_expr = self.parse_index_expression(left_expr)?;
                }
                _ => return Some(left_expr),
            }
        }

        Some(left_expr)
    }

    fn parse_list_literal(&mut self) -> Option<Expression> {
        let mut elements = Vec::new();

        if self.peek_token_is(&Token::RBracket) {
            self.next_token(); // move to ]
            return Some(Expression::List(elements));
        }

        self.next_token(); // skip [
        elements.push(self.parse_expression(Precedence::Lowest)?);

        // Check for list comprehension
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
            let iterable = self.parse_expression(Precedence::Lowest)?;
            
            let mut condition = None;
            if self.peek_token_is(&Token::If) {
                self.next_token();
                self.next_token();
                condition = Some(Box::new(self.parse_expression(Precedence::Lowest)?));
            }

            if !self.expect_peek(Token::RBracket) {
                return None;
            }
            return Some(Expression::ListComprehension {
                element: Box::new(elements.remove(0)),
                variable,
                iterable: Box::new(iterable),
                condition,
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

        Some(Expression::List(elements))
    }

    fn parse_dictionary_literal(&mut self) -> Option<Expression> {
        let mut pairs = Vec::new();

        if self.peek_token_is(&Token::RBrace) {
            self.next_token(); // move to }
            return Some(Expression::Dictionary(pairs));
        }

        self.next_token(); // skip {
        
        loop {
            let key = self.parse_expression(Precedence::Lowest)?;
            if !self.expect_peek(Token::Colon) {
                return None;
            }
            self.next_token(); // skip :
            let value = self.parse_expression(Precedence::Lowest)?;
            pairs.push((key, value));
            
            if self.peek_token_is(&Token::RBrace) {
                self.next_token();
                break;
            }
            if !self.expect_peek(Token::Comma) {
                return None;
            }
            self.next_token(); // skip ,
        }
        
        Some(Expression::Dictionary(pairs))
    }

    fn parse_index_expression(&mut self, left: Expression) -> Option<Expression> {
        let index = self.parse_expression(Precedence::Lowest)?;

        if !self.expect_peek(Token::RBracket) {
            return None;
        }

        Some(Expression::Index {
            object: Box::new(left),
            index: Box::new(index),
        })
    }

    fn parse_prefix_expression(&mut self) -> Option<Expression> {
        let operator = match self.cur_token {
            Token::Minus => PrefixOperator::Minus,
            Token::Not => PrefixOperator::Not,
            Token::Tilde => PrefixOperator::BitwiseNot,
            _ => return None,
        };
        
        self.next_token();
        let right = self.parse_expression(Precedence::Prefix)?;
        
        Some(Expression::Prefix { operator, right: Box::new(right) })
    }

    fn parse_infix_expression(&mut self, left: Expression) -> Option<Expression> {
        let operator = match self.cur_token {
            Token::Plus => InfixOperator::Plus,
            Token::Minus => InfixOperator::Minus,
            Token::Star => InfixOperator::Multiply,
            Token::Slash => InfixOperator::Divide,
            Token::Percent => InfixOperator::Modulo,
            Token::Equal => InfixOperator::Equal,
            Token::NotEqual => InfixOperator::NotEqual,
            Token::LessThan => InfixOperator::LessThan,
            Token::GreaterThan => InfixOperator::GreaterThan,
            Token::LessThanOrEqual => InfixOperator::LessThanOrEqual,
            Token::GreaterThanOrEqual => InfixOperator::GreaterThanOrEqual,
            Token::At => InfixOperator::MatrixMultiply,
            Token::DoubleStar => InfixOperator::Power,
            Token::In => InfixOperator::In,
            Token::NotIn => InfixOperator::NotIn,
            Token::And => InfixOperator::And,
            Token::Or => InfixOperator::Or,
            Token::Ampersand => InfixOperator::BitwiseAnd,
            Token::Pipe => InfixOperator::BitwiseOr,
            Token::Caret => InfixOperator::BitwiseXor,
            Token::ShiftLeft => InfixOperator::ShiftLeft,
            Token::ShiftRight => InfixOperator::ShiftRight,
            _ => return None,
        };

        let precedence = self.cur_precedence();
        self.next_token();
        let right = self.parse_expression(precedence)?;

        Some(Expression::Infix { left: Box::new(left), operator, right: Box::new(right) })
    }
    
    fn parse_call_expression(&mut self, function: Expression) -> Option<Expression> {
        let arguments = self.parse_call_arguments()?;
        Some(Expression::Call { function: Box::new(function), arguments })
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
        // .name
        // Expect Identifier
        match &self.peek_token {
            Token::Identifier(_) => {
                self.next_token();
                match &self.cur_token {
                    Token::Identifier(name) => Some(Expression::Get { object: Box::new(object), name: name.clone() }),
                    _ => None,
                }
            },
            _ => {
                self.errors.push(format!("Expected Identifier after '.', got {:?}", self.peek_token));
                None
            }
        }
    }

    fn peek_precedence(&self) -> Precedence {
        match self.peek_token {
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
        // Simple check. For Identifier, we check variant.
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
                self.errors.push(format!("Expected {:?}, got {:?}", t, self.peek_token));
                false
            }
        }
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
}
