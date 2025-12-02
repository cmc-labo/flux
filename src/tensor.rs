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
    
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        // ndarray dot product
        // For 2D matrices: (m x k) . (k x n) -> (m x n)
        // ndarray::ArrayBase::dot supports 1D and 2D.
        
        // We need to ensure they are 2D for now for simplicity?
        // Or just let ndarray handle it.
        // However, ArrayD dot is not directly available as a simple method for all dimensions.
        // We might need to cast to 2D if we want strict matrix multiplication.
        // But let's try to use the `dot` method if available on ArrayD or cast.
        
        // ArrayD doesn't have `dot`. We need to downcast to Array2 if we want 2D matmul.
        // Or use `linalg::general_mat_mul`?
        
        // Let's assume 2D for now.
        let a = self.inner.clone().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("LHS must be 2D for matmul: {}", e))?;
        let b = other.inner.clone().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("RHS must be 2D for matmul: {}", e))?;
            
        let res = a.dot(&b);
        Ok(Tensor { inner: res.into_dyn() })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
