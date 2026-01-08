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
    For,
    In,
    Not,

    // Operators
    Plus,
    Minus,
    Star,
    DoubleStar,
    Slash,
    Percent,
    Assign,
    Equal, // ==
    NotEqual, // !=
    LessThan, // <
    GreaterThan, // >
    LessThanOrEqual, // <=
    GreaterThanOrEqual, // >=
    At, // @
    Dot, // .
    LBracket, // [
    RBracket, // ]
    
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
