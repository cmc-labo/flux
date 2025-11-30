#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Identifier(String),

    // Keywords
    Def,
    Return,
    If,
    Else,
    While,
    Let,
    Print,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Assign,
    Equal, // ==
    
    // Delimiters
    LParen,
    RParen,
    Colon,
    Comma,
    
    // Structure
    Newline,
    Indent,
    Dedent,
    EOF,
    
    // Error
    Illegal(String),
}
