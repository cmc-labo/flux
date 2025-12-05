#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let { name: String, value: Expression },
    Return(Option<Expression>),
    Expression(Expression),
    FunctionDef { name: String, params: Vec<String>, body: Block },
    If { condition: Expression, consequence: Block, alternative: Option<Block> },
    While { condition: Expression, body: Block },
    Print(Expression),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Identifier(String),
    Integer(i64),
    Float(f64),
    String(String),
    Prefix { operator: PrefixOperator, right: Box<Expression> },
    Infix { left: Box<Expression>, operator: InfixOperator, right: Box<Expression> },
    Call { function: Box<Expression>, arguments: Vec<Expression> },
    Get { object: Box<Expression>, name: String }, // obj.name
}

#[derive(Debug, PartialEq, Clone)]
pub enum PrefixOperator {
    Minus,
    Not, // ! or not
}

#[derive(Debug, PartialEq, Clone)]
pub enum InfixOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    Assign, // For now, maybe treated as statement? But Python allows x = y = 1. Let's keep it simple: assignment is a statement in our MVP.
    MatrixMultiply,
}
