use miette::{Diagnostic, SourceSpan};
use thiserror::Error;
use crate::span::Span;

#[derive(Error, Debug, Diagnostic)]
pub enum FluxError {
    #[error("Parse Error: {message}")]
    #[diagnostic(code(flux::parser::error))]
    ParseError {
        message: String,
        #[label("{message}")]
        span: SourceSpan,
    },

    #[error("Runtime Error: {message}")]
    #[diagnostic(code(flux::interpreter::error))]
    RuntimeError {
        message: String,
        #[label("{message}")]
        span: SourceSpan,
    },

    #[error("Type Error: {message}")]
    #[diagnostic(code(flux::type_checker::error))]
    TypeError {
        message: String,
        #[label("{message}")]
        span: SourceSpan,
    },
}

impl FluxError {
    pub fn new_parse(message: String, span: Span) -> Self {
        FluxError::ParseError {
            message,
            span: span.into(),
        }
    }

    pub fn new_runtime(message: String, span: Span) -> Self {
        FluxError::RuntimeError {
            message,
            span: span.into(),
        }
    }

    pub fn new_type(message: String, span: Span) -> Self {
        FluxError::TypeError {
            message,
            span: span.into(),
        }
    }
}
