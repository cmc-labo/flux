use ndarray::{Array, ArrayD, IxDyn};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub inner: ArrayD<f64>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let array = Array::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| format!("Failed to create tensor: {}", e))?;
        Ok(Tensor { inner: array })
    }
    
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        // ndarray handles shape checking and broadcasting (if we wanted)
        // For strict element-wise addition with same shape:
        if self.inner.shape() != other.inner.shape() {
             return Err(format!("Shape mismatch: {:?} vs {:?}", self.inner.shape(), other.inner.shape()));
        }
        
        let res = &self.inner + &other.inner;
        Ok(Tensor { inner: res })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
