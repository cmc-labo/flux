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
    Elif,
    While,
    Let,
    Print,
    And,
    Or,
    For,
    In,
    NotIn,
    Not,
    Break,
    Continue,
    Import,

    // Operators
    Plus,
    Minus,
    Star,
    DoubleStar,
    Slash,
    Percent,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    PercentAssign,
    DoubleStarAssign, // **=
    AmpersandAssign, // &=
    PipeAssign, // |=
    CaretAssign, // ^=
    ShiftLeftAssign, // <<=
    ShiftRightAssign, // >>=
    Equal, // ==
    NotEqual, // !=
    LessThan, // <
    GreaterThan, // >
    LessThanOrEqual, // <=
    GreaterThanOrEqual, // >=
    At, // @
    Dot, // .
    Ampersand, // &
    Pipe, // |
    Caret, // ^
    Tilde, // ~
    ShiftLeft, // <<
    ShiftRight, // >>
    LBracket, // [
    RBracket, // ]
    LBrace, // {
    RBrace, // }
    
    // Delimiters
    LParen,
    RParen,
    Colon,
    Comma,
    Arrow, // ->
    
    // Structure
    Newline,
    Indent,
    Dedent,
    EOF,
    
    // Error
    Illegal(String),
}
