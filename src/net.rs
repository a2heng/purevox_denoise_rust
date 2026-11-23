
use crate::block::{BlockTrace, GateBlock};
use crate::meta::{LayerCfg, Meta};
use crate::weights::{WeightError, Weights};

pub struct Encoder {
    pub layers: Vec<GateBlock>,
}

impl Encoder {
    pub fn from_weights(w: &Weights, meta: &Meta) -> Result<Self, WeightError> {
        let layers = meta.enc.iter().enumerate()
            .map(|(i, c)| build(w, &format!("encoder.layers.{i}"), c, meta.arch.tfa_kt, meta.arch.bn_eps))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut h = x.to_vec();
        let mut skips = Vec::with_capacity(self.layers.len());
        for blk in &self.layers {
            h = blk.forward(&h, t);
            skips.push(h.clone());
        }
        (h, skips)
    }

    pub fn forward_trace(&self, x: &[f32], t: usize)
        -> (Vec<f32>, Vec<Vec<f32>>, Vec<BlockTrace>)
    {
        let mut h = x.to_vec();
        let mut skips = Vec::with_capacity(self.layers.len());
        let mut traces = Vec::with_capacity(self.layers.len());
        for blk in &self.layers {
            let tr = blk.forward_trace(&h, t);
            h = tr.out.clone();
            skips.push(h.clone());
            traces.push(tr);
        }
        (h, skips, traces)
    }
}

pub struct Decoder {
    pub layers: Vec<GateBlock>,
}

impl Decoder {
    pub fn from_weights(w: &Weights, meta: &Meta) -> Result<Self, WeightError> {
        let layers = meta.dec.iter().enumerate()
            .map(|(i, c)| build(w, &format!("decoder.layers.{i}"), c, meta.arch.tfa_kt, meta.arch.bn_eps))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &[f32], t: usize, skips: &[Vec<f32>]) -> Vec<f32> {
        let n_enc = self.layers.len();
        let mut h = x.to_vec();
        for (i, blk) in self.layers.iter().enumerate() {
            h = blk.forward(&h, t);
            if i < n_enc - 1 {
                let s = &skips[n_enc - 2 - i];
                assert_eq!(s.len(), h.len());
                for (a, b) in h.iter_mut().zip(s) { *a += b; }
            }
        }
        h
    }

    pub fn forward_stages(&self, x: &[f32], t: usize, skips: &[Vec<f32>])
        -> (Vec<f32>, Vec<BlockTrace>, Vec<Vec<f32>>)
    {
        let n_enc = self.layers.len();
        let mut h = x.to_vec();
        let mut traces = Vec::with_capacity(n_enc);
        let mut stages = Vec::with_capacity(n_enc);
        for (i, blk) in self.layers.iter().enumerate() {
            let tr = blk.forward_trace(&h, t);
            h = tr.out.clone();
            if i < n_enc - 1 {
                let s = &skips[n_enc - 2 - i];
                assert_eq!(s.len(), h.len());
                for (a, b) in h.iter_mut().zip(s) { *a += b; }
            }
            traces.push(tr);
            stages.push(h.clone());
        }
        (h, traces, stages)
    }
}

fn build(w: &Weights, prefix: &str, c: &LayerCfg, tfa_kt: usize, bn_eps: f32)
    -> Result<GateBlock, WeightError>
{
    GateBlock::from_weights(w, prefix, c.in_width, c.stride, c.pad_w, c.use_deconv,
                            c.tfa, tfa_kt, bn_eps)
}
