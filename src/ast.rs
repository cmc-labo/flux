#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let { name: String, value: Expression },
    Return(Option<Expression>),
    Expression(Expression),
    FunctionDef { name: String, params: Vec<String>, body: Block },
    If { condition: Expression, consequence: Block, alternative: Option<Block> },
    While { condition: Expression, body: Block },
    For { variable: String, iterable: Expression, body: Block },
    Print(Vec<Expression>),
    IndexAssign { object: Expression, index: Expression, value: Expression },
    Break,
    Continue,
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
    List(Vec<Expression>),
    Index { object: Box<Expression>, index: Box<Expression> },
    ListComprehension { element: Box<Expression>, variable: String, iterable: Box<Expression>, condition: Option<Box<Expression>> },
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
}
