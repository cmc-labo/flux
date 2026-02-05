use crate::span::Span;

#[derive(Debug, PartialEq, Clone)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    Null,
    List(Box<Type>),
    Dictionary(Box<Type>, Box<Type>),
    Set(Box<Type>),
    Tensor,
    Any,
    Function(Vec<Type>, Box<Type>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum StatementKind {
    Let { 
        names: Vec<String>, 
        value: Expression,
        type_hint: Option<Type>,
    },
    Return(Option<Expression>),
    Expression(Expression),
    FunctionDef { 
        name: String, 
        params: Vec<(String, Option<Type>)>, 
        body: Block,
        return_type: Option<Type>,
    },
    If { condition: Expression, consequence: Block, elif_branches: Vec<(Expression, Block)>, alternative: Option<Block> },
    While { condition: Expression, body: Block },
    For { variables: Vec<String>, iterable: Expression, body: Block },
    Print(Vec<Expression>),
    IndexAssign { object: Expression, index: Expression, value: Expression },
    Break,
    Continue,
    Import { path: String, alias: Option<String> },
    Assert { condition: Expression, message: Option<Expression> },
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression {
    pub kind: ExpressionKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Clone)]
pub enum FStringPart {
    Literal(String),
    Expression(Box<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum ExpressionKind {
    Identifier(String),
    Integer(i64),
    Float(f64),
    String(String),
    Prefix { operator: PrefixOperator, right: Box<Expression> },
    Infix { left: Box<Expression>, operator: InfixOperator, right: Box<Expression> },
    Call { function: Box<Expression>, arguments: Vec<Expression> },
    MethodCall { object: Box<Expression>, method: String, arguments: Vec<Expression> },
    Get { object: Box<Expression>, name: String }, // obj.name
    List(Vec<Expression>),
    Dictionary(Vec<(Expression, Expression)>),
    Set(Vec<Expression>),
    Index { object: Box<Expression>, index: Box<Expression> },
    ListComprehension { element: Box<Expression>, variable: String, iterable: Box<Expression>, condition: Option<Box<Expression>> },
    SetComprehension { element: Box<Expression>, variable: String, iterable: Box<Expression>, condition: Option<Box<Expression>> },
    DictComprehension { key: Box<Expression>, value: Box<Expression>, variable: String, iterable: Box<Expression>, condition: Option<Box<Expression>> },
    Slice { start: Option<Box<Expression>>, end: Option<Box<Expression>>, step: Option<Box<Expression>> },
    Ternary { condition: Box<Expression>, consequence: Box<Expression>, alternative: Box<Expression> },
    Lambda { params: Vec<String>, body: Box<Expression> },
    FString(Vec<FStringPart>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum PrefixOperator {
    Minus,
    Not, // ! or not
    BitwiseNot, // ~
}

#[derive(Debug, PartialEq, Clone)]
pub enum InfixOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    And,
    Or,
    Power,
    In,
    NotIn,
    MatrixMultiply,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}
