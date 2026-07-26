# -*- coding: utf-8 -*-
"""CDE-ARDiff: covariate-conditioned dynamic-anchor residual diffusion."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_t(t, d):
    h = d // 2
    f = torch.exp(torch.arange(h, device=t.device) * (-math.log(10000) / max(h - 1, 1)))
    e = torch.cat([torch.sin(t[:, None].float() * f[None]), torch.cos(t[:, None].float() * f[None])], 1)
    return F.pad(e, (0, d - e.shape[1])) if e.shape[1] < d else e


class PriceEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (2, 3), padding=(0, 1)), nn.GELU(),
            nn.Conv2d(16, 32, (2, 3), padding=(0, 1)), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 6)), nn.Flatten(),
            nn.Linear(32 * 6, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))


class CovariateEncoder(nn.Module):
    def __init__(self, cfg, feat_dim):
        super().__init__()
        d = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d), nn.LayerNorm(d),
        )
    def forward(self, feat):
        return self.net(feat)


class DynamicAnchor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, k = cfg.d_model, cfg.pred_len, 5
        self.h = h; self.k = k
        self.weight_net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, k * h))
        self.corr = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(2 * d, h))
        init = torch.tensor(getattr(cfg, "anchor_init_weights", (0.30, 0.30, 0.20, 0.10, 0.10)), dtype=torch.float32)
        self.init_logits = nn.Parameter(torch.log(init.clamp_min(1e-4)).view(1, k, 1))
        self.corr_gate = nn.Parameter(torch.tensor(float(getattr(cfg, "anchor_correction_init", -1.5))))
    def candidates(self, matrix, bd, bw):
        recent_mean = matrix[:, :3, :].mean(1)
        seasonal_mean = matrix[:, 3:, :].mean(1)
        recent_median = matrix[:, :3, :].median(1).values
        return torch.stack([bd, bw, recent_mean, seasonal_mean, recent_median], 1)
    def forward(self, z_price, z_cov, matrix, bd, bw, residual_scale):
        cand = self.candidates(matrix, bd, bw)
        logits = self.weight_net(torch.cat([z_price, z_cov], 1)).view(-1, self.k, self.h) + self.init_logits
        w = F.softmax(logits, 1)
        base = (cand * w).sum(1)
        raw = self.corr(torch.cat([z_price, z_cov], 1))
        corr = residual_scale.view(1, -1) * torch.tanh(raw / residual_scale.view(1, -1))
        anchor = base + torch.sigmoid(self.corr_gate) * corr
        return anchor, base, w


class RegimeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.net = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.reg = nn.Linear(d, 3); self.tr = nn.Linear(d, 3); self.vol = nn.Linear(d, 3)
        self.guide = nn.Linear(d, d)
    def forward(self, z):
        h = self.net(z)
        rl, tl, vl = self.reg(h), self.tr(h), self.vol(h)
        rp, tp, vp = F.softmax(rl, 1), F.softmax(tl, 1), F.softmax(vl, 1)
        return rl, tl, vl, rp, tp, vp, self.guide(h)


class ResidualDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg.d_model, cfg.pred_len
        self.t_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.net = nn.Sequential(
            nn.Linear(h + h + 4 * d, 3 * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * d, h),
        )
    def forward(self, x, t, anchor, z_price, z_cov, guide):
        te = self.t_proj(sinusoidal_t(t, z_price.shape[1]))
        return self.net(torch.cat([x, anchor, z_price, z_cov, guide, te], 1))


class CDEARDiff(nn.Module):
    def __init__(self, cfg, feat_dim, residual_scale=None, class_weights=None):
        super().__init__()
        self.cfg = cfg
        self.price = PriceEncoder(cfg); self.cov = CovariateEncoder(cfg, feat_dim)
        self.anchor = DynamicAnchor(cfg); self.regime = RegimeHead(cfg); self.denoiser = ResidualDenoiser(cfg)
        self.gate = nn.Sequential(nn.Linear(2 * cfg.d_model + 3, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.pred_len), nn.Sigmoid())
        self.gate_bias = nn.Parameter(torch.full((cfg.pred_len,), float(getattr(cfg, "diffusion_gate_bias", -3.0))))
        self.register_buffer("residual_scale", torch.as_tensor(torch.ones(cfg.pred_len) if residual_scale is None else residual_scale, dtype=torch.float32).clamp_min(1e-3))
        self.register_buffer("horizon_w", torch.tensor(cfg.horizon_loss_weights, dtype=torch.float32))
        cw = class_weights or (torch.ones(3), torch.ones(3), torch.ones(3))
        self.register_buffer("reg_w", torch.as_tensor(cw[0], dtype=torch.float32)); self.register_buffer("tri_w", torch.as_tensor(cw[1], dtype=torch.float32)); self.register_buffer("vol_w", torch.as_tensor(cw[2], dtype=torch.float32))
    def encode(self, matrix, feat):
        zp, zc = self.price(matrix), self.cov(feat)
        rl, tl, vl, rp, tp, vp, guide = self.regime(torch.cat([zp, zc], 1))
        return zp, zc, (rl, tl, vl, rp, tp, vp, guide)
    def alpha_bar(self, t):
        beta0, beta1 = 1e-4, 0.02
        beta = beta0 + (beta1 - beta0) * t.float().view(-1, 1) / max(self.cfg.diffusion_steps - 1, 1)
        return torch.clamp((1 - beta) ** (t.float().view(-1, 1) + 1), 1e-5, 0.9999)
    def forward_train(self, matrix, feat, y, labels, bd, bw, t, noise):
        zp, zc, cls = self.encode(matrix, feat); rl, tl, vl, rp, tp, vp, guide = cls
        anchor, base, weights = self.anchor(zp, zc, matrix, bd, bw, self.residual_scale)
        residual = torch.clamp((y - anchor.detach()) / self.residual_scale.view(1, -1), -self.cfg.norm_clip, self.cfg.norm_clip)
        a = self.alpha_bar(t); sa, so = torch.sqrt(a), torch.sqrt(1 - a)
        xt = torch.clamp(sa * residual + so * noise, -self.cfg.ddim_clip, self.cfg.ddim_clip)
        v_target = sa * noise - so * residual
        v_pred = self.denoiser(xt, t, anchor.detach(), zp, zc, guide)
        x0 = torch.clamp(sa * xt - so * v_pred, -self.cfg.norm_clip, self.cfg.norm_clip)
        diff = anchor + self.residual_scale.view(1, -1) * torch.tanh(x0)
        gate_in = torch.cat([zp, zc, torch.stack([rp.max(1).values, tp.max(1).values, vp.max(1).values], 1)], 1)
        gate = torch.sigmoid(self.gate(gate_in) + self.gate_bias.view(1, -1))
        pred = anchor + gate * (diff - anchor)
        reg, tr, vol = labels
        cls_loss = F.cross_entropy(rl, reg, weight=self.reg_w) + F.cross_entropy(tl, tr, weight=self.tri_w) + F.cross_entropy(vl, vol, weight=self.vol_w)
        hw = self.horizon_w.view(1, -1)
        anchor_loss = (F.smooth_l1_loss(anchor, y, reduction="none") * hw).mean()
        final_loss = (F.smooth_l1_loss(pred, y, reduction="none") * hw).mean()
        robust = (torch.log1p(torch.abs(pred - y) / self.residual_scale.view(1, -1)) * hw).mean()
        shape = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], y[:, 1:] - y[:, :-1])
        v_loss = F.mse_loss(v_pred, v_target); x0_loss = F.smooth_l1_loss(x0, residual)
        gate_reg = gate.mean()
        losses = {"anchor": anchor_loss, "final": final_loss, "robust": robust, "shape": shape, "v": v_loss, "x0": x0_loss, "cls": cls_loss, "gate": gate_reg}
        total = (self.cfg.anchor_weight * anchor_loss + self.cfg.final_weight * final_loss + self.cfg.robust_weight * robust + self.cfg.shape_weight * shape + self.cfg.diffusion_weight * v_loss + self.cfg.x0_weight * x0_loss + self.cfg.cls_weight * cls_loss + self.cfg.gate_reg_weight * gate_reg)
        return total, losses, pred, anchor, base, weights, cls
    @torch.no_grad()
    def predict(self, matrix, feat, bd, bw, samples=10):
        zp, zc, cls = self.encode(matrix, feat); rl, tl, vl, rp, tp, vp, guide = cls
        anchor, base, weights = self.anchor(zp, zc, matrix, bd, bw, self.residual_scale)
        preds=[]
        for _ in range(samples):
            x = torch.randn_like(anchor)
            steps = torch.linspace(self.cfg.diffusion_steps - 1, 0, self.cfg.ddim_steps, device=matrix.device).long()
            for i in range(len(steps)-1):
                t = torch.full((len(matrix),), int(steps[i].item()), device=matrix.device, dtype=torch.long)
                a = self.alpha_bar(t); sa, so = torch.sqrt(a), torch.sqrt(1 - a)
                v = self.denoiser(x, t, anchor, zp, zc, guide)
                x = torch.clamp(sa * x - so * v, -self.cfg.norm_clip, self.cfg.norm_clip)
            diff = anchor + self.residual_scale.view(1, -1) * torch.tanh(x)
            gate_in = torch.cat([zp, zc, torch.stack([rp.max(1).values, tp.max(1).values, vp.max(1).values], 1)], 1)
            gate = torch.sigmoid(self.gate(gate_in) + self.gate_bias.view(1, -1))
            preds.append(anchor + gate * (diff - anchor))
        return torch.stack(preds), anchor, base, weights, cls
