
use crate::block::{CTfa, GateBlock};
use crate::dpgrnn::DpGrnn;
use crate::erb::Erb;
use crate::memory::{BottleneckMemory, BANDS};
use crate::meta::Meta;
use crate::model::{mask_act, Denoiser};
use crate::stft::Spectrogram;

#[derive(Debug, Clone)]
pub struct StreamCache {
    pub enc: Vec<Option<Vec<f32>>>,
    pub dec: Vec<Option<Vec<f32>>>,
    pub tfa: Vec<Vec<f32>>,
    pub inter: Vec<Vec<f32>>,
    pub mem_g: Vec<f32>,
    pub mem_s: Vec<f32>,
}

impl StreamCache {
    pub fn zeros(meta: &Meta) -> Self {
        let mk = |plan: &Vec<crate::meta::PlanEntry>| -> Vec<Option<Vec<f32>>> {
            plan.iter().map(|e| e.shape.as_ref().map(|s| vec![0.0f32; s.iter().product()])).collect()
        };
        Self {
            enc: mk(&meta.cache.enc),
            dec: mk(&meta.cache.dec),
            tfa: meta.cache.tfa.iter().map(|e| vec![0.0f32; e.shape.as_ref().unwrap().iter().product()]).collect(),
            inter: meta.cache.inter.iter().map(|e| vec![0.0f32; e.shape.as_ref().unwrap().iter().product()]).collect(),
            mem_g: vec![0.0f32; meta.cache.mem[0].shape.as_ref().unwrap().iter().product()],
            mem_s: vec![0.0f32; meta.cache.mem[1].shape.as_ref().unwrap().iter().product()],
        }
    }

    pub fn total(&self) -> usize {
        let sz = |v: &Vec<Option<Vec<f32>>>| v.iter().flatten().map(|x| x.len()).sum::<usize>();
        sz(&self.enc) + sz(&self.dec)
            + self.tfa.iter().map(|v| v.len()).sum::<usize>()
            + self.inter.iter().map(|v| v.len()).sum::<usize>()
            + self.mem_g.len() + self.mem_s.len()
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total());
        for v in self.enc.iter().chain(self.dec.iter()).flatten() { out.extend_from_slice(v); }
        for v in &self.tfa { out.extend_from_slice(v); }
        for v in &self.inter { out.extend_from_slice(v); }
        out.extend_from_slice(&self.mem_g);
        out.extend_from_slice(&self.mem_s);
        out
    }

    pub fn from_flat(&mut self, flat: &[f32]) {
        assert_eq!(flat.len(), self.total(), "cache 长度不符");
        let mut o = 0;
        for v in self.enc.iter_mut().chain(self.dec.iter_mut()).flatten() {
            let n = v.len();
            v.copy_from_slice(&flat[o..o + n]); o += n;
        }
        for v in self.tfa.iter_mut() { let n = v.len(); v.copy_from_slice(&flat[o..o + n]); o += n; }
        for v in self.inter.iter_mut() { let n = v.len(); v.copy_from_slice(&flat[o..o + n]); o += n; }
        let n = self.mem_g.len();
        self.mem_g.copy_from_slice(&flat[o..o + n]); o += n;
        let n = self.mem_s.len();
        self.mem_s.copy_from_slice(&flat[o..o + n]);
    }
}

impl CTfa {
    pub fn stream(&self, x: &[f32], w: usize, cache: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let ch = self.channels();
        assert_eq!(x.len(), ch * w);
        assert_eq!(cache.len(), ch * self.frames);
        let (mut fg, _, _) = self.fgate.forward(x, 1, 1, w);
        crate::ops::act::sigmoid_slice(&mut fg);
        let mut h1 = vec![0.0f32; x.len()];
        for i in 0..x.len() { h1[i] = x[i] * fg[i]; }
        // zt = mean(h1²) → (C,1,1)
        let mut seq = vec![0.0f32; ch * self.kt]; // (C, kt, 1)
        for c in 0..ch {
            let mut acc = 0.0f32;
            for k in 0..w { let v = h1[c * w + k]; acc += v * v; }
            seq[c * self.kt + self.frames] = acc / w as f32;
        }
        for c in 0..ch {
            for j in 0..self.frames {
                seq[c * self.kt + j] = cache[c * self.frames + j];
            }
        }
        let (mut g, _, _) = self.gate.forward(&seq, 1, self.kt, 1);
        crate::ops::act::sigmoid_slice(&mut g);
        let mut out = vec![0.0f32; x.len()];
        for c in 0..ch {
            let gv = g[c];
            for k in 0..w { out[c * w + k] = h1[c * w + k] * gv; }
        }
        let mut new_cache = vec![0.0f32; ch * self.frames];
        for c in 0..ch {
            for j in 0..self.frames {
                new_cache[c * self.frames + j] = seq[c * self.kt + 1 + j];
            }
        }
        (out, new_cache)
    }
}

impl GateBlock {
    pub fn stream(&self, x: &[f32], in_w: usize, conv_cache: Option<&[f32]>, tfa_cache: Option<&[f32]>)
        -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)
    {
        let h0 = self.pconv.forward(x, 1, 1, in_w).0;
        let b1 = self.bn1.forward(&h0, self.out_ch, 1, in_w);
        let mut hs = b1;
        crate::ops::act::hswish_slice(&mut hs);

        let (mut padded, tp, new_conv) = if self.kt == 1 {
            (hs.clone(), 1usize, None)
        } else {
            let frames = self.kt - 1;
            let cache = conv_cache.expect("kt>1 必须给 conv cache");
            assert_eq!(cache.len(), self.out_ch * frames * in_w);
            let mut p = vec![0.0f32; self.out_ch * self.kt * in_w];
            for c in 0..self.out_ch {
                p[c * self.kt * in_w..c * self.kt * in_w + frames * in_w]
                    .copy_from_slice(&cache[c * frames * in_w..(c + 1) * frames * in_w]);
                p[c * self.kt * in_w + frames * in_w..(c + 1) * self.kt * in_w]
                    .copy_from_slice(&hs[c * in_w..(c + 1) * in_w]);
            }
            let mut nc = vec![0.0f32; self.out_ch * frames * in_w];
            for c in 0..self.out_ch {
                nc[c * frames * in_w..(c + 1) * frames * in_w]
                    .copy_from_slice(&p[c * self.kt * in_w + in_w..(c + 1) * self.kt * in_w]);
            }
            (std::mem::take(&mut p), self.kt, Some(nc))
        };
        let (y, _, ow) = self.dconv.forward(&padded, 1, tp, in_w);
        padded.clear();
        assert_eq!(ow, self.out_width, "流式频率宽度不符");
        let b2 = self.bn2.forward(&y, self.out_ch, 1, ow);
        let (h, new_tfa) = match &self.tfa {
            Some(tfa) => {
                let (o, nc) = tfa.stream(&b2, ow, tfa_cache.expect("tfa cache"));
                (o, Some(nc))
            }
            None => (b2, None),
        };
        let mut out = h;
        if self.in_ch == self.out_ch && self.in_width == self.out_width {
            for i in 0..out.len() { out[i] += x[i]; }
        }
        (out, new_conv, new_tfa)
    }
}

impl DpGrnn {
    pub fn stream(&self, x: &[f32], f: usize, inter: &mut [f32]) -> Vec<f32> {
        let c = self.input;
        assert_eq!(x.len(), c * f);
        assert_eq!(inter.len(), f * self.hidden);
        let (dw, _, _) = self.intra_dw.forward(x, 1, 1, f);
        let (pw, _, _) = self.intra_pw.forward(&dw, 1, 1, f);
        // (F,C) → DyT → (C,F)
        let mut perm = vec![0.0f32; f * c];
        for ci in 0..c {
            for fi in 0..f { perm[fi * c + ci] = pw[ci * f + fi]; }
        }
        let nl = self.intra_norm.forward(&perm);
        let mut intra = vec![0.0f32; x.len()];
        for ci in 0..c {
            for fi in 0..f { intra[ci * f + fi] = x[ci * f + fi] + nl[fi * c + ci]; }
        }
        let mut seq = vec![0.0f32; f * c];
        for fi in 0..f {
            for ci in 0..c { seq[fi * c + ci] = intra[ci * f + fi]; }
        }
        let ry = self.inter_rnn.step(&seq, inter);
        let lin = self.inter_fc.forward(&ry, f);
        let mut flat = vec![0.0f32; f * c];
        for fi in 0..f {
            for ci in 0..c { flat[fi * c + ci] = lin[fi * c + ci]; }
        }
        let nn = self.inter_norm.forward(&flat);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            for fi in 0..f { out[ci * f + fi] = intra[ci * f + fi] + nn[fi * c + ci]; }
        }
        out
    }
}

impl BottleneckMemory {
    pub fn stream(&self, x: &[f32], f: usize, gstate: &mut [f32], sstate: &mut [f32]) -> Vec<f32> {
        let c = self.channels;
        assert_eq!(x.len(), c * f);
        assert_eq!(gstate.len(), self.tgru.layers.len() * self.hidden_of_global());
        assert_eq!(sstate.len(), BANDS * self.bhid);

        let mut tfc = vec![0.0f32; f * c];
        for ti in 0..1 {
            for fi in 0..f {
                for ci in 0..c { tfc[(ti * f + fi) * c + ci] = x[ci * f + fi]; }
            }
        }
        let ln = self.ln.forward(&tfc);
        let h0 = vec![0.0f32; 2 * self.fhid];
        let (fy, _) = self.fgru.forward(&ln, 1, f, &h0);
        let fp = self.fproj.forward(&fy, f);
        let mut cur = x.to_vec();
        for ci in 0..c {
            for fi in 0..f { cur[ci * f + fi] += fp[fi * c + ci]; }
        }

        cur = self.proto_frame(&cur, f);

        let fb = f / BANDS;
        let mut g = vec![0.0f32; BANDS * c];
        for ci in 0..c {
            for bi in 0..BANDS {
                let mut acc = 0.0f32;
                for k in 0..fb { acc += cur[ci * f + bi * fb + k]; }
                g[bi * c + ci] = acc / fb as f32;
            }
        }
        let sp = self.sgru.step(&g, sstate);
        let sp = self.sproj.forward(&sp, BANDS);
        let mut out = vec![0.0f32; cur.len()];
        for ci in 0..c {
            for bi in 0..BANDS {
                let gate = sp[bi * c + ci].tanh();
                for k in 0..fb { let i = ci * f + bi * fb + k; out[i] = cur[i] * (1.0 + gate); }
            }
        }

        let mut gm = vec![0.0f32; c];
        for ci in 0..c {
            let mut acc = 0.0f32;
            for fi in 0..f { acc += out[ci * f + fi]; }
            gm[ci] = acc / f as f32;
        }
        let ty = self.tgru.step(&gm, gstate);
        let tp = self.tproj.forward(&ty, 1);
        for ci in 0..c {
            let gate = tp[ci].tanh();
            for fi in 0..f { out[ci * f + fi] *= 1.0 + gate; }
        }
        out
    }
}

#[derive(Debug, Default)]
pub struct HopTrace {
    pub feat: Vec<f32>,
    pub enc: Vec<Vec<f32>>,
    pub dp: Vec<Vec<f32>>,
    pub pre_mem: Vec<f32>,
    pub mem: Vec<f32>,
    pub dec: Vec<Vec<f32>>,
    pub logit: Vec<f32>,
}

impl Denoiser {
    pub fn stream_hop_trace(&self, frame: &[f32], cache: &mut StreamCache) -> (Vec<f32>, HopTrace) {
        let bins = self.meta.erb.n_freq;
        let spec = Spectrogram { frames: 1, bins, data: frame.to_vec() };
        let logmag = Erb::logmag(&spec);
        let feat = self.erb.bm(&logmag, 1);
        let mut tr = HopTrace { feat: feat.clone(), ..Default::default() };

        let mut x = feat;
        let mut tfa_i = 0usize;
        for (i, blk) in self.encoder.layers.iter().enumerate() {
            let (y, nc, nt) = blk.stream(&x, blk.in_width, cache.enc[i].as_deref(),
                                         Some(cache.tfa[tfa_i].as_slice()));
            cache.enc[i] = nc;
            if let Some(t) = nt { cache.tfa[tfa_i] = t; }
            tfa_i += 1;
            tr.enc.push(y.clone());
            x = y;
        }
        let f = self.meta.arch.enc_out_w[self.meta.arch.enc_out_w.len() - 1];
        let top = x.clone();
        let mut y = top.clone();
        for (k, dp) in self.dps.iter().enumerate() {
            y = dp.stream(&y, f, &mut cache.inter[k]);
            tr.dp.push(y.clone());
        }
        let mut pre = y.clone();
        for i in 0..pre.len() { pre[i] += top[i]; }
        tr.pre_mem = pre.clone();
        let mem = self.mem.stream(&pre, f, &mut cache.mem_g, &mut cache.mem_s);
        tr.mem = mem.clone();

        let mut h = mem;
        let n = self.decoder.layers.len();
        let mut tfa_i = meta_enc_tfa(&self.meta);
        for (i, blk) in self.decoder.layers.iter().enumerate() {
            let state = if blk.tfa.is_some() {
                let s = Some(cache.tfa[tfa_i].as_slice());
                tfa_i += 1;
                s
            } else { None };
            let (mut y, nc, nt) = blk.stream(&h, blk.in_width, cache.dec[i].as_deref(), state);
            cache.dec[i] = nc;
            if let Some(t) = nt { cache.tfa[tfa_i - 1] = t; }
            if i < n - 1 {
                let sk = &tr.enc[n - 2 - i];
                for (a, b) in y.iter_mut().zip(sk) { *a += b; }
            }
            tr.dec.push(y.clone());
            h = y;
        }
        tr.logit = h.clone();
        let u = self.erb.bs(&h, 1);
        let mask = mask_act(&u, self.meta.mask.floor_db, self.meta.mask.depth_gain);
        let mut out = vec![0.0f32; 2 * bins];
        for ch in 0..2 {
            for b in 0..bins { out[ch * bins + b] = frame[ch * bins + b] * mask[b]; }
        }
        (out, tr)
    }

    pub fn stream_hop(&self, frame: &[f32], cache: &mut StreamCache, ctrl: Option<(f32, f32)>)
        -> Vec<f32>
    {
        if ctrl.is_none() {
            return self.stream_hop_trace(frame, cache).0;
        }
        let bins = self.meta.erb.n_freq;
        let spec = Spectrogram { frames: 1, bins, data: frame.to_vec() };
        let logmag = Erb::logmag(&spec);
        let mut x = self.erb.bm(&logmag, 1);
        let mut skips: Vec<Vec<f32>> = Vec::with_capacity(self.encoder.layers.len());
        let mut tfa_i = 0usize;
        for (i, blk) in self.encoder.layers.iter().enumerate() {
            let (y, nc, nt) = blk.stream(&x, blk.in_width, cache.enc[i].as_deref(),
                                         Some(cache.tfa[tfa_i].as_slice()));
            cache.enc[i] = nc;
            if let Some(t) = nt { cache.tfa[tfa_i] = t; }
            tfa_i += 1;
            skips.push(y.clone());
            x = y;
        }
        let f = self.meta.arch.enc_out_w[self.meta.arch.enc_out_w.len() - 1];
        let top = x.clone();
        let mut y = top.clone();
        for (k, dp) in self.dps.iter().enumerate() {
            y = dp.stream(&y, f, &mut cache.inter[k]);
        }
        let mut pre = vec![0.0f32; y.len()];
        for i in 0..pre.len() { pre[i] = y[i] + top[i]; }
        let mem = self.mem.stream(&pre, f, &mut cache.mem_g, &mut cache.mem_s);
        let mut h = mem;
        let n = self.decoder.layers.len();
        let mut ti = meta_enc_tfa(&self.meta);
        for (i, blk) in self.decoder.layers.iter().enumerate() {
            let state = if blk.tfa.is_some() {
                let s = Some(cache.tfa[ti].as_slice());
                ti += 1;
                s
            } else { None };
            let (mut yy, nc, nt) = blk.stream(&h, blk.in_width, cache.dec[i].as_deref(), state);
            cache.dec[i] = nc;
            if let Some(t) = nt { cache.tfa[ti - 1] = t; }
            if i < n - 1 {
                let sk = &skips[n - 2 - i];
                for (a, b) in yy.iter_mut().zip(sk) { *a += b; }
            }
            h = yy;
        }
        let u = self.erb.bs(&h, 1);
        let (a, sp) = ctrl.unwrap();
        let mask = mask_act(&u, Some(a), sp);
        let mut out = vec![0.0f32; 2 * bins];
        for ch in 0..2 {
            for b in 0..bins { out[ch * bins + b] = frame[ch * bins + b] * mask[b]; }
        }
        out
    }
}

impl BottleneckMemory {
    fn hidden_of_global(&self) -> usize {
        self.tgru.out_dim()
    }

    fn proto_frame(&self, x: &[f32], f: usize) -> Vec<f32> {
        let c = self.channels;
        let d = self.dim;
        let mut pooled = vec![0.0f32; c];
        for ci in 0..c {
            let mut acc = 0.0f32;
            for fi in 0..f { acc += x[ci * f + fi]; }
            pooled[ci] = acc / f as f32;
        }
        let q = self.pq.forward(&pooled, 1);
        let scale = 1.0 / (d as f32).sqrt();
        let mut logits = vec![0.0f32; self.slots];
        for si in 0..self.slots {
            let mut a = 0.0f32;
            for k in 0..d { a += q[k] * self.slots_w[si * d + k]; }
            logits[si] = a * scale;
        }
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in logits.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
        let mut rd = vec![0.0f32; d];
        for k in 0..d {
            let mut a = 0.0f32;
            for si in 0..self.slots { a += (logits[si] / sum) * self.slots_w[si * d + k]; }
            rd[k] = a;
        }
        let y = self.pv.forward(&rd, 1);
        let mut out = vec![0.0f32; x.len()];
        for ci in 0..c {
            let g = 1.0 + y[ci].tanh();
            for fi in 0..f { out[ci * f + fi] = x[ci * f + fi] * g; }
        }
        out
    }
}

fn meta_enc_tfa(meta: &crate::meta::Meta) -> usize {
    meta.enc.iter().filter(|c| c.tfa).count()
}
