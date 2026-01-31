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

    pub fn min(&self) -> f64 {
        self.inner.fold(f64::INFINITY, |a, &b| a.min(b))
    }

    pub fn max(&self) -> f64 {
        self.inner.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    pub fn var(&self) -> f64 {
        let mean = self.inner.mean().unwrap_or(0.0);
        self.inner.fold(0.0, |acc, &x| acc + (x - mean).powi(2)) / self.inner.len() as f64
    }

    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    pub fn abs(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.abs()) }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.sqrt()) }
    }

    pub fn exp(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.exp()) }
    }

    pub fn log(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.ln()) }
    }

    pub fn prod(&self) -> f64 {
        self.inner.product()
    }

    pub fn all(&self) -> bool {
        self.inner.iter().all(|&x| x != 0.0)
    }

    pub fn any(&self) -> bool {
        self.inner.iter().any(|&x| x != 0.0)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
