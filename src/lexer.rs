use crate::token::Token;
use std::iter::Peekable;
use std::str::Chars;

pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<Chars<'a>>,
    indent_stack: Vec<usize>,
    pending_tokens: Vec<Token>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input,
            chars: input.chars().peekable(),
            indent_stack: vec![0],
            pending_tokens: Vec::new(),
        };
        // Initial check for indentation if the file doesn't start with empty lines?
        // For simplicity, we assume standard flow or handle first token logic in next_token
        lexer
    }

    pub fn next_token(&mut self) -> Token {
        if !self.pending_tokens.is_empty() {
            return self.pending_tokens.remove(0);
        }

        self.skip_whitespace();

        if let Some(&ch) = self.chars.peek() {
            match ch {
                'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
                '0'..='9' => self.read_number(),
                '"' => self.read_string(),
                '=' => {
                    self.chars.next();
                    if let Some(&'=') = self.chars.peek() {
                        self.chars.next();
                        Token::Equal
                    } else {
                        Token::Assign
                    }
                }
                '+' => { self.chars.next(); Token::Plus }
                '-' => { self.chars.next(); Token::Minus }
                '*' => { self.chars.next(); Token::Star }
                '/' => { self.chars.next(); Token::Slash }
                '%' => { self.chars.next(); Token::Percent }
                '@' => { self.chars.next(); Token::At }
                '(' => { self.chars.next(); Token::LParen }
                ')' => { self.chars.next(); Token::RParen }
                ':' => { self.chars.next(); Token::Colon }
                ',' => { self.chars.next(); Token::Comma }
                '.' => { self.chars.next(); Token::Dot }
                '\n' => self.handle_newline(),
                _ => {
                    self.chars.next();
                    Token::Illegal(ch.to_string())
                }
            }
        } else {
            // EOF
            // If we have indentation, we need to dedent back to 0
            if self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                return Token::Dedent;
            }
            Token::EOF
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    fn handle_newline(&mut self) -> Token {
        self.chars.next(); // consume '\n'
        
        // Calculate indentation level
        let mut indent_level = 0;
        while let Some(&ch) = self.chars.peek() {
            if ch == ' ' {
                indent_level += 1;
                self.chars.next();
            } else if ch == '\t' {
                indent_level += 4; // Assume 4 spaces for tab
                self.chars.next();
            } else {
                break;
            }
        }

        // If line is empty or comment (not handled yet), just continue? 
        // For now, if we hit another newline, we recurse or loop.
        if let Some(&'\n') = self.chars.peek() {
            return self.handle_newline();
        }
        
        // Check EOF after newline
        if self.chars.peek().is_none() {
             // EOF logic will handle dedents
             return self.next_token();
        }

        let current_indent = *self.indent_stack.last().unwrap_or(&0); // Safety
        
        if indent_level > current_indent {
            self.indent_stack.push(indent_level);
            self.pending_tokens.push(Token::Indent);
        } else if indent_level < current_indent {
             while let Some(&top) = self.indent_stack.last() {
                if top > indent_level {
                    self.pending_tokens.push(Token::Dedent);
                    self.indent_stack.pop();
                } else {
                    break;
                }
            }
        }
        
        Token::Newline
    }

    fn read_identifier(&mut self) -> Token {
        let mut literal = String::new();
        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                literal.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        match literal.as_str() {
            "def" => Token::Def,
            "return" => Token::Return,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "let" => Token::Let,
            "print" => Token::Print,
            "and" => Token::And,
            "or" => Token::Or,
            _ => Token::Identifier(literal),
        }
    }

    fn read_number(&mut self) -> Token {
        let mut literal = String::new();
        let mut is_float = false;
        
        while let Some(&ch) = self.chars.peek() {
            if ch.is_digit(10) {
                literal.push(ch);
                self.chars.next();
            } else if ch == '.' && !is_float {
                is_float = true;
                literal.push(ch);
                self.chars.next();
            } else {
                break;
            }
        }
        
        if is_float {
            Token::Float(literal.parse().unwrap_or(0.0))
        } else {
            Token::Integer(literal.parse().unwrap_or(0))
        }
    }

    fn read_string(&mut self) -> Token {
        self.chars.next(); // skip "
        let mut literal = String::new();
        while let Some(&ch) = self.chars.peek() {
            if ch == '"' {
                break;
            }
            literal.push(ch);
            self.chars.next();
        }
        self.chars.next(); // consume closing "
        Token::String(literal)
    }
}
