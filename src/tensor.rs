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
        if self.inner.shape() != other.inner.shape() {
             return Err(format!("Shape mismatch: {:?} vs {:?}", self.inner.shape(), other.inner.shape()));
        }
        let res = &self.inner + &other.inner;
        Ok(Tensor { inner: res })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.inner.shape() != other.inner.shape() {
             return Err(format!("Shape mismatch: {:?} vs {:?}", self.inner.shape(), other.inner.shape()));
        }
        let res = &self.inner - &other.inner;
        Ok(Tensor { inner: res })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.inner.shape() != other.inner.shape() {
             return Err(format!("Shape mismatch: {:?} vs {:?}", self.inner.shape(), other.inner.shape()));
        }
        let res = &self.inner * &other.inner;
        Ok(Tensor { inner: res })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.inner.shape() != other.inner.shape() {
             return Err(format!("Shape mismatch: {:?} vs {:?}", self.inner.shape(), other.inner.shape()));
        }
        let res = &self.inner / &other.inner;
        Ok(Tensor { inner: res })
    }
    
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.inner.clone().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("LHS must be 2D for matmul: {}", e))?;
        let b = other.inner.clone().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| format!("RHS must be 2D for matmul: {}", e))?;
            
        let res = a.dot(&b);
        Ok(Tensor { inner: res.into_dyn() })
    }

    pub fn transpose(&self) -> Tensor {
        Tensor { inner: self.inner.t().to_owned() }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, String> {
        let res = self.inner.clone().into_shape_with_order(IxDyn(&shape))
            .map_err(|e| format!("Reshape failed: {}", e))?;
        Ok(Tensor { inner: res.into_dyn() })
    }

    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }

    pub fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Tensor { inner: ArrayD::zeros(IxDyn(&shape)) }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Tensor { inner: ArrayD::ones(IxDyn(&shape)) }
    }

    pub fn rand(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<f64> = (0..total_size).map(|_| rng.gen()).collect();
        Tensor::new(data, shape).unwrap()
    }

    pub fn add_scalar(&self, scalar: f64) -> Self {
        Tensor { inner: &self.inner + scalar }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Self {
        Tensor { inner: &self.inner - scalar }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Self {
        Tensor { inner: &self.inner * scalar }
    }

    pub fn div_scalar(&self, scalar: f64) -> Self {
        Tensor { inner: &self.inner / scalar }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
