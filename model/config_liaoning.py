# -*- coding: utf-8 -*-
from dataclasses import dataclass
import os

@dataclass
class Config:
    data_path: str = r"D:\Desktop\研究生\课题组\研究课题-电价预测\文献及数据集\辽宁数据集\GridPulse_Liaoning.csv"
    sheet_name: str = "Sheet1"
    time_col: str = "Time"
    target_col: str = "Real-time price"
    output_dir: str = r"D:\Desktop\山东电价预测\RHC_CDE_ARDiff\outputs_liaoning"
    market: str = "liaoning"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    pred_len: int = 24
    rows: int = 6
    cols: int = 24
    patch_len: int = 6
    warmup_rows: int = 0
    d_model: int = 80
    dropout: float = 0.20
    diffusion_steps: int = 300
    ddim_steps: int = 10
    mc_samples: int = 5
    norm_clip: float = 6.0
    ddim_clip: float = 8.0
    residual_quantile: float = 0.92
    anchor_init_weights: tuple = (0.25, 0.45, 0.20, 0.05, 0.05)
    anchor_correction_init: float = -1.4
    diffusion_gate_bias: float = -3.5
    anchor_weight: float = 1.25
    final_weight: float = 1.10
    robust_weight: float = 0.35
    shape_weight: float = 0.15
    diffusion_weight: float = 0.16
    x0_weight: float = 0.10
    cls_weight: float = 0.10
    gate_reg_weight: float = 0.015
    val_metric_bonus: float = 0.03
    post_calibrate: bool = False
    refit_train_val_epochs: int = 0
    tree_auxiliary: bool = False
    horizon_loss_weights: tuple = (1,1,1,1,1,1,1.05,1.05,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.15,1.15,1.15,1.15,1.1,1.05,1.05)
    batch_size: int = 96
    epochs: int = 300
    patience: int = 60
    lr: float = 3e-4
    weight_decay: float = 3e-4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 2024
    num_workers: int = 0
    use_gpu: bool = True
    selected_historical_features = ("Real-time tie-line power","Real-time load","Photovoltaic output","wind output","Hydro/pumped-storage output","renewable total","non-market generation","nuclear output","2m air temperature","weather code","2m dew-point temperature","Sea-level pressure","surface pressure","Total precipitation","rainfall","snowfall","2m relative humidity","apparent temperature","10m gust speed","10m wind direction","10m wind speed","100m wind speed")
    use_future_known_covariates: bool = True
    selected_future_known_features = selected_historical_features
    def ensure_dirs(self): os.makedirs(self.output_dir, exist_ok=True)
