use std::collections::HashMap;
use std::fs;
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors: {0}")]
    Safe(#[from] safetensors::SafeTensorError),
    #[error("tensor not found: {0}")]
    Missing(String),
    #[error("tensor {name} is not f32 (dtype={dtype:?})")]
    Dtype { name: String, dtype: safetensors::Dtype },
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct Weights {
    pub map: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let bytes = fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut map = HashMap::new();
        for name in st.names() {
            let t = st.tensor(name)?;
            if t.dtype() != safetensors::Dtype::F32 {
                continue; // BN num_batches_tracked 等整数元数据，不进 f32 权重表
            }
            let raw = t.data();
            let mut data = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            map.insert(name.to_string(), Tensor { shape: t.shape().to_vec(), data });
        }
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, WeightError> {
        self.map.get(name).ok_or_else(|| WeightError::Missing(name.to_string()))
    }
}
