use ndarray::{Array, ArrayD, IxDyn};
use std::fmt;
use crate::object::Object;

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
        match (self.inner.ndim(), other.inner.ndim()) {
            (2, 2) => {
                let a = self.inner.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b = other.inner.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                Ok(Tensor { inner: a.dot(&b).into_dyn() })
            }
            (1, 2) => {
                let a = self.inner.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
                let b = other.inner.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                Ok(Tensor { inner: a.dot(&b).into_dyn() })
            }
            (2, 1) => {
                let a = self.inner.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b = other.inner.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
                Ok(Tensor { inner: a.dot(&b).into_dyn() })
            }
            (1, 1) => {
                let a = self.inner.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
                let b = other.inner.clone().into_dimensionality::<ndarray::Ix1>().unwrap();
                // dot of two 1D arrays returns a scalar
                let res = a.dot(&b);
                Ok(Tensor { inner: Array::from_elem(IxDyn(&[]), res) })
            }
            _ => Err(format!("matmul supported for 1D/2D only, got {}D and {}D", self.inner.ndim(), other.inner.ndim())),
        }
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

    pub fn flatten(&self) -> Tensor {
        Tensor { inner: self.inner.clone().into_shape_with_order(self.inner.len()).unwrap().into_dyn() }
    }

    pub fn ceil(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.ceil()) }
    }

    pub fn floor(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.floor()) }
    }

    pub fn round(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.round()) }
    }

    pub fn isinf(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| if x.is_infinite() { 1.0 } else { 0.0 }) }
    }

    pub fn isnan(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| if x.is_nan() { 1.0 } else { 0.0 }) }
    }

    pub fn log2(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.log2()) }
    }

    pub fn log10(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.log10()) }
    }

    pub fn copy(&self) -> Tensor {
        Tensor { inner: self.inner.clone() }
    }

    pub fn argmax(&self, axis: Option<usize>) -> Object {
        use ndarray::Axis;
        if let Some(a) = axis {
            let res = self.inner.map_axis(Axis(a), |view| {
                view.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as f64).unwrap_or(0.0)
            });
            Object::Tensor(Tensor { inner: res.into_dyn() })
        } else {
            let idx = self.inner.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64).unwrap_or(0);
            Object::Integer(idx)
        }
    }

    pub fn argmin(&self, axis: Option<usize>) -> Object {
        use ndarray::Axis;
        if let Some(a) = axis {
            let res = self.inner.map_axis(Axis(a), |view| {
                view.iter().enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as f64).unwrap_or(0.0)
            });
            Object::Tensor(Tensor { inner: res.into_dyn() })
        } else {
            let idx = self.inner.iter().enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64).unwrap_or(0);
            Object::Integer(idx)
        }
    }

    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor, String> {
        let mut shape = self.inner.shape().to_vec();
        if let Some(a) = axis {
            if a >= shape.len() { return Err(format!("Axis {} out of bounds for shape {:?}", a, shape)); }
            if shape[a] != 1 { return Err(format!("Cannot squeeze axis {} with size {}", a, shape[a])); }
            shape.remove(a);
        } else {
            shape.retain(|&s| s != 1);
        }
        let res = self.inner.clone().into_shape_with_order(IxDyn(&shape))
            .map_err(|e| format!("Squeeze failed: {}", e))?;
        Ok(Tensor { inner: res.into_dyn() })
    }

    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor, String> {
        let mut shape = self.inner.shape().to_vec();
        if axis > shape.len() { return Err(format!("Axis {} out of bounds for shape {:?}", axis, shape)); }
        shape.insert(axis, 1);
        let res = self.inner.clone().into_shape_with_order(IxDyn(&shape))
            .map_err(|e| format!("Unsqueeze failed: {}", e))?;
        Ok(Tensor { inner: res.into_dyn() })
    }

    pub fn abs(&self) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.abs()) }
    }

    pub fn diag(&self, k: i32) -> Result<Tensor, String> {
        if self.inner.ndim() != 2 {
            return Err("diag() requires a 2D tensor".to_string());
        }
        let rows = self.inner.shape()[0] as i32;
        let cols = self.inner.shape()[1] as i32;
        
        let mut diag_elements = Vec::new();
        for i in 0..rows {
            let j = i + k;
            if j >= 0 && j < cols {
                diag_elements.push(self.inner[[i as usize, j as usize]]);
            }
        }
        
        let n = diag_elements.len();
        Ok(Tensor { inner: Array::from_shape_vec(IxDyn(&[n]), diag_elements).unwrap() })
    }

    pub fn trace(&self) -> Result<f64, String> {
        if self.inner.ndim() != 2 {
            return Err("trace() requires a 2D tensor".to_string());
        }
        let n = std::cmp::min(self.inner.shape()[0], self.inner.shape()[1]);
        let mut sum = 0.0;
        for i in 0..n {
            sum += self.inner[[i, i]];
        }
        Ok(sum)
    }

    pub fn clip(&self, min: f64, max: f64) -> Tensor {
        Tensor { inner: self.inner.mapv(|x| x.clamp(min, max)) }
    }

    pub fn norm(&self, p: f64) -> f64 {
        if p == 2.0 {
            self.inner.iter().map(|&x| x * x).sum::<f64>().sqrt()
        } else if p.is_infinite() {
            self.inner.iter().map(|&x| x.abs()).fold(0.0, f64::max)
        } else {
            self.inner.iter().map(|&x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
        }
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
