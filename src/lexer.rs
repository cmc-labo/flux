use crate::token::Token;
use crate::span::Span;
use std::iter::Peekable;
use std::str::Chars;

pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<Chars<'a>>,
    pos: usize,
    indent_stack: Vec<usize>,
    pending_tokens: Vec<(Token, Span)>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            input,
            chars: input.chars().peekable(),
            pos: 0,
            indent_stack: vec![0],
            pending_tokens: Vec::new(),
        }
    }

    fn next_char(&mut self) -> Option<char> {
        let ch = self.chars.next()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    pub fn next_token(&mut self) -> (Token, Span) {
        if !self.pending_tokens.is_empty() {
            return self.pending_tokens.remove(0);
        }

        self.skip_whitespace();

        let start = self.pos;
        if let Some(&ch) = self.chars.peek() {
            match ch {
                'a'..='z' | 'A'..='Z' | '_' => self.read_identifier(),
                '0'..='9' => self.read_number(),
                '"' => self.read_string(),
                '=' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::Equal, Span::new(start, self.pos - start))
                    } else {
                        (Token::Assign, Span::new(start, self.pos - start))
                    }
                }
                '!' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::NotEqual, Span::new(start, self.pos - start))
                    } else {
                        (Token::Illegal("!".to_string()), Span::new(start, self.pos - start))
                    }
                }
                '<' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::LessThanOrEqual, Span::new(start, self.pos - start))
                    } else if let Some(&'<') = self.chars.peek() {
                        self.next_char();
                        if let Some(&'=') = self.chars.peek() {
                            self.next_char();
                            (Token::ShiftLeftAssign, Span::new(start, self.pos - start))
                        } else {
                            (Token::ShiftLeft, Span::new(start, self.pos - start))
                        }
                    } else {
                        (Token::LessThan, Span::new(start, self.pos - start))
                    }
                }
                '>' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::GreaterThanOrEqual, Span::new(start, self.pos - start))
                    } else if let Some(&'>') = self.chars.peek() {
                        self.next_char();
                        if let Some(&'=') = self.chars.peek() {
                            self.next_char();
                            (Token::ShiftRightAssign, Span::new(start, self.pos - start))
                        } else {
                            (Token::ShiftRight, Span::new(start, self.pos - start))
                        }
                    } else {
                        (Token::GreaterThan, Span::new(start, self.pos - start))
                    }
                }
                '&' => {
                    self.next_char();
                    if let Some(&'&') = self.chars.peek() {
                        self.next_char();
                        (Token::And, Span::new(start, self.pos - start))
                    } else if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::AmpersandAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Ampersand, Span::new(start, self.pos - start))
                    }
                }
                '|' => {
                    self.next_char();
                    if let Some(&'|') = self.chars.peek() {
                        self.next_char();
                        (Token::Or, Span::new(start, self.pos - start))
                    } else if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::PipeAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Pipe, Span::new(start, self.pos - start))
                    }
                }
                '+' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::PlusAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Plus, Span::new(start, self.pos - start))
                    }
                }
                '-' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::MinusAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Minus, Span::new(start, self.pos - start))
                    }
                }
                '*' => {
                    self.next_char();
                    if let Some(&'*') = self.chars.peek() {
                        self.next_char();
                        if let Some(&'=') = self.chars.peek() {
                            self.next_char();
                            (Token::DoubleStarAssign, Span::new(start, self.pos - start))
                        } else {
                            (Token::DoubleStar, Span::new(start, self.pos - start))
                        }
                    } else if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::StarAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Star, Span::new(start, self.pos - start))
                    }
                }
                '/' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::SlashAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Slash, Span::new(start, self.pos - start))
                    }
                }
                '%' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::PercentAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Percent, Span::new(start, self.pos - start))
                    }
                }
                '^' => {
                    self.next_char();
                    if let Some(&'=') = self.chars.peek() {
                        self.next_char();
                        (Token::CaretAssign, Span::new(start, self.pos - start))
                    } else {
                        (Token::Caret, Span::new(start, self.pos - start))
                    }
                }
                '~' => { self.next_char(); (Token::Tilde, Span::new(start, self.pos - start)) }
                '@' => { self.next_char(); (Token::At, Span::new(start, self.pos - start)) }
                '(' => { self.next_char(); (Token::LParen, Span::new(start, self.pos - start)) }
                ')' => { self.next_char(); (Token::RParen, Span::new(start, self.pos - start)) }
                ':' => { self.next_char(); (Token::Colon, Span::new(start, self.pos - start)) }
                ',' => { self.next_char(); (Token::Comma, Span::new(start, self.pos - start)) }
                '.' => { self.next_char(); (Token::Dot, Span::new(start, self.pos - start)) }
                '[' => { self.next_char(); (Token::LBracket, Span::new(start, self.pos - start)) }
                ']' => { self.next_char(); (Token::RBracket, Span::new(start, self.pos - start)) }
                '{' => { self.next_char(); (Token::LBrace, Span::new(start, self.pos - start)) }
                '}' => { self.next_char(); (Token::RBrace, Span::new(start, self.pos - start)) }
                '#' => {
                    self.skip_comment();
                    self.next_token()
                }
                '\n' => self.handle_newline(),
                _ => {
                    self.next_char();
                    (Token::Illegal(ch.to_string()), Span::new(start, self.pos - start))
                }
            }
        } else {
            if self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                return (Token::Dedent, Span::new(self.pos, 0));
            }
            (Token::EOF, Span::new(self.pos, 0))
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.next_char();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(&ch) = self.chars.peek() {
            if ch == '\n' {
                break;
            }
            self.next_char();
        }
    }

    fn handle_newline(&mut self) -> (Token, Span) {
        let start = self.pos;
        self.next_char(); // consume '\n'
        
        let mut indent_level = 0;
        while let Some(&ch) = self.chars.peek() {
            if ch == ' ' {
                indent_level += 1;
                self.next_char();
            } else if ch == '\t' {
                indent_level += 4;
                self.next_char();
            } else {
                break;
            }
        }

        if let Some(&'\n') = self.chars.peek() {
            return self.handle_newline();
        }
        
        if self.chars.peek().is_none() {
             return self.next_token();
        }

        let current_indent = *self.indent_stack.last().unwrap_or(&0);
        let span = Span::new(start, self.pos - start);
        
        if indent_level > current_indent {
            self.indent_stack.push(indent_level);
            self.pending_tokens.push((Token::Indent, span));
        } else if indent_level < current_indent {
             while let Some(&top) = self.indent_stack.last() {
                if top > indent_level {
                    self.pending_tokens.push((Token::Dedent, span));
                    self.indent_stack.pop();
                } else {
                    break;
                }
            }
        }
        
        (Token::Newline, span)
    }

    fn read_identifier(&mut self) -> (Token, Span) {
        let start = self.pos;
        let mut literal = String::new();
        while let Some(&ch) = self.chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                literal.push(ch);
                self.next_char();
            } else {
                break;
            }
        }
        let span = Span::new(start, self.pos - start);
        let token = match literal.as_str() {
            "def" => Token::Def,
            "return" => Token::Return,
            "if" => Token::If,
            "else" => Token::Else,
            "elif" => Token::Elif,
            "while" => Token::While,
            "let" => Token::Let,
            "print" => Token::Print,
            "and" => Token::And,
            "or" => Token::Or,
            "for" => Token::For,
            "in" => Token::In,
            "not" => {
                let remaining: String = self.chars.clone().collect();
                if remaining.starts_with(" in ") || remaining.starts_with(" in\n") || remaining.starts_with(" in\t") {
                    self.next_char(); // " "
                    self.next_char(); // "i"
                    self.next_char(); // "n"
                    Token::NotIn
                } else {
                    Token::Not
                }
            },
            "break" => Token::Break,
            "continue" => Token::Continue,
            "import" => Token::Import,
            _ => Token::Identifier(literal),
        };
        (token, span)
    }

    fn read_number(&mut self) -> (Token, Span) {
        let start = self.pos;
        let mut literal = String::new();
        let mut is_float = false;
        
        while let Some(&ch) = self.chars.peek() {
            if ch.is_digit(10) {
                literal.push(ch);
                self.next_char();
            } else if ch == '.' && !is_float {
                is_float = true;
                literal.push(ch);
                self.next_char();
            } else {
                break;
            }
        }
        
        let span = Span::new(start, self.pos - start);
        let token = if is_float {
            Token::Float(literal.parse().unwrap_or(0.0))
        } else {
            Token::Integer(literal.parse().unwrap_or(0))
        };
        (token, span)
    }

    fn read_string(&mut self) -> (Token, Span) {
        let start = self.pos;
        self.next_char(); // skip "
        let mut literal = String::new();
        while let Some(ch) = self.next_char() {
            if ch == '"' {
                break;
            }
            if ch == '\\' {
                if let Some(next_ch) = self.next_char() {
                    match next_ch {
                        'n' => literal.push('\n'),
                        't' => literal.push('\t'),
                        '\\' => literal.push('\\'),
                        '"' => literal.push('"'),
                        _ => {
                            literal.push('\\');
                            literal.push(next_ch);
                        }
                    }
                } else {
                    literal.push('\\');
                }
            } else {
                literal.push(ch);
            }
        }
        let span = Span::new(start, self.pos - start);
        (Token::String(literal), span)
    }
}

