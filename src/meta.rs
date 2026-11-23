use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Meta {
    pub model: String,
    #[serde(default)]
    pub checkpoint: String,
    pub stft: StftCfg,
    pub erb: ErbCfg,
    pub cache: CacheCfg,
    pub arch: Arch,
    pub mask: Mask,
    pub enc: Vec<LayerCfg>,
    pub dec: Vec<LayerCfg>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LayerCfg {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kt: usize,
    pub kf: usize,
    pub stride: usize,
    pub in_width: usize,
    pub out_width: usize,
    pub pad_w: usize,
    pub pad_t: usize,
    pub use_deconv: bool,
    pub tfa: bool,
    #[serde(default)]
    pub groups: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct StftCfg {
    pub n_fft: usize,
    pub hop: usize,
    pub win: usize,
    pub fs: u32,
    pub center: bool,
    pub pad_mode: String,
}

#[derive(Debug, Deserialize)]
pub struct ErbCfg {
    pub low: usize,
    pub high: usize,
    pub f_comp: usize,
    pub n_freq: usize,
    pub w_shape: Vec<usize>,
    pub w_avg_shape: Vec<usize>,
    pub w_up_shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct CacheCfg {
    pub total: usize,
    pub enc: Vec<PlanEntry>,
    pub dec: Vec<PlanEntry>,
    pub tfa: Vec<PlanEntry>,
    pub inter: Vec<PlanEntry>,
    pub hold: Vec<PlanEntry>,
    pub mem: Vec<PlanEntry>,
}

#[derive(Debug, Deserialize)]
pub struct PlanEntry {
    pub offset: usize,
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
pub struct Arch {
    pub enc_ch: Vec<usize>,
    pub dec_ch: Vec<usize>,
    pub enc_out_w: Vec<usize>,
    pub tfa_kt: usize,
    pub tfa_frames: usize,
    pub intra_kf: usize,
    pub bn_eps: f32,
    pub dp_hid: usize,
    pub n_dp: usize,
    pub mem_layers: usize,
    pub mem_hid: usize,
    pub mem_fhid: usize,
    pub mem_bands: usize,
    pub mem_bhid: usize,
    pub mem_slots: usize,
    pub mem_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct Mask {
    pub floor_db: Option<f32>,
    pub depth_gain: f32,
}
