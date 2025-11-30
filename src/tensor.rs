use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }
    
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }
        
        let new_data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Tensor::new(new_data, self.shape.clone()))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor({:?})", self.data)
    }
}
