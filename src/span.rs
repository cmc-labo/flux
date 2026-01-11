use miette::SourceSpan;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Span {
    pub offset: usize,
    pub len: usize,
}

impl Span {
    pub fn new(offset: usize, len: usize) -> Self {
        Span { offset, len }
    }

    pub fn join(self, other: Span) -> Span {
        let start = self.offset.min(other.offset);
        let end = (self.offset + self.len).max(other.offset + other.len);
        Span {
            offset: start,
            len: end - start,
        }
    }
}

impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        SourceSpan::new(span.offset.into(), span.len.into())
    }
}
