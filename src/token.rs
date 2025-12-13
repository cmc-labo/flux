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
    And,
    Or,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Assign,
    Equal, // ==
    NotEqual, // !=
    LessThan, // <
    GreaterThan, // >
    At, // @
    Dot, // .
    
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
