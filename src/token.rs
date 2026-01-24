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
    Assert,

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
    Semicolon, // ;
    
    // Structure
    Newline,
    Indent,
    Dedent,
    EOF,
    
    // Error
    Illegal(String),
}

impl Token {
    pub fn to_infix_operator(&self) -> Option<crate::ast::InfixOperator> {
        match self {
            Token::Plus => Some(crate::ast::InfixOperator::Plus),
            Token::Minus => Some(crate::ast::InfixOperator::Minus),
            Token::Star => Some(crate::ast::InfixOperator::Multiply),
            Token::Slash => Some(crate::ast::InfixOperator::Divide),
            Token::Percent => Some(crate::ast::InfixOperator::Modulo),
            Token::Equal => Some(crate::ast::InfixOperator::Equal),
            Token::NotEqual => Some(crate::ast::InfixOperator::NotEqual),
            Token::LessThan => Some(crate::ast::InfixOperator::LessThan),
            Token::GreaterThan => Some(crate::ast::InfixOperator::GreaterThan),
            Token::LessThanOrEqual => Some(crate::ast::InfixOperator::LessThanOrEqual),
            Token::GreaterThanOrEqual => Some(crate::ast::InfixOperator::GreaterThanOrEqual),
            Token::At => Some(crate::ast::InfixOperator::MatrixMultiply),
            Token::DoubleStar => Some(crate::ast::InfixOperator::Power),
            Token::In => Some(crate::ast::InfixOperator::In),
            Token::NotIn => Some(crate::ast::InfixOperator::NotIn),
            Token::And => Some(crate::ast::InfixOperator::And),
            Token::Or => Some(crate::ast::InfixOperator::Or),
            Token::Ampersand => Some(crate::ast::InfixOperator::BitwiseAnd),
            Token::Pipe => Some(crate::ast::InfixOperator::BitwiseOr),
            Token::Caret => Some(crate::ast::InfixOperator::BitwiseXor),
            Token::ShiftLeft => Some(crate::ast::InfixOperator::ShiftLeft),
            Token::ShiftRight => Some(crate::ast::InfixOperator::ShiftRight),
            _ => None,
        }
    }
}
