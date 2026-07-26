# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import Dataset, DataLoader


def segment_id(hour):
    if hour < 6:
        return 0
    if hour < 9:
        return 1
    if hour < 12:
        return 2
    if hour < 15:
        return 3
    if hour < 18:
        return 4
    if hour < 21:
        return 5
    return 6


def time_features(dt):
    hour = dt.dt.hour.values.astype(np.float32)
    dow = dt.dt.dayofweek.values.astype(np.float32)
    month = dt.dt.month.values.astype(np.float32)
    weekend = (dow >= 5).astype(np.float32)
    return {
        "hour": hour,
        "is_weekend": weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }


def preprocess_liaoning_missing_values(df, cfg):
    df = df.copy()
    missing_cols = [
        cfg.target_col,
        "Real-time tie-line power",
        "Photovoltaic output",
        "wind output",
        "Hydro/pumped-storage output",
    ]
    for col in missing_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col == cfg.target_col:
            df.loc[df[col] == -9999, col] = np.nan
        df[col] = df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
    return df


def load_data_strict(cfg):
    path = cfg.data_path
    with open(path, 'rb') as f:
        header = f.read(4)
    if header[:4] == b'PK\x03\x04':
        df = pd.read_excel(path, sheet_name=cfg.sheet_name)
    elif path.endswith('.csv'):
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='gbk')
    else:
        df = pd.read_excel(path, sheet_name=cfg.sheet_name)
    df = normalize_market_columns(df, cfg)

    def parse_time_series(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().all():
            return parsed

        def convert_market_time(x):
            x = str(x).strip().replace("：", ":")
            date_part, hour_part = x.split()
            if "." in date_part:
                y, m, d = [int(v) for v in date_part.split(".")]
            else:
                y, m, d = [int(v) for v in date_part.replace("/", "-").split("-")]
            hh, mm = [int(v) for v in hour_part.split(":")[:2]]
            base = pd.Timestamp(y, m, d)
            if hh == 24:
                return base + pd.Timedelta(days=1)
            return base + pd.Timedelta(hours=hh, minutes=mm)

        return series.apply(convert_market_time)

    df[cfg.time_col] = parse_time_series(df[cfg.time_col])

    df = df.sort_values(cfg.time_col).reset_index(drop=True)
    if getattr(cfg, "market", "").lower() == "liaoning":
        df = preprocess_liaoning_missing_values(df, cfg)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    raw = df[cfg.target_col].astype(float).values
    train_n = int(len(raw) * cfg.train_ratio)
    yj = PowerTransformer(method="yeo-johnson", standardize=False)
    yj.fit(raw[:train_n].reshape(-1, 1))
    price_yj = yj.transform(raw.reshape(-1, 1)).reshape(-1)

    train_yj = price_yj[:train_n]
    margin = 0.10 * (train_yj.max() - train_yj.min() + 1e-8)
    clip_min = float(train_yj.min() - margin)
    clip_max = float(train_yj.max() + margin)
    df["price_yj"] = price_yj
    return df, yj, clip_min, clip_max


def inverse_yj(yj, arr, clip_min, clip_max):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, clip_min, clip_max)
    shape = arr.shape
    return yj.inverse_transform(arr.reshape(-1, 1)).reshape(shape)


def add_causal_features(df, cfg):
    df = df.copy()
    tf = time_features(df[cfg.time_col])
    for k, v in tf.items():
        df[k] = v
    seg = df[cfg.time_col].dt.hour.apply(segment_id).values
    df["segment_id"] = seg
    for s in range(7):
        df[f"segment_{s}"] = (seg == s).astype(np.float32)

    shifted = df["price_yj"].shift(1)
    roll_cols = []
    for w in [3, 6, 12, 24, 48, 168]:
        stats = {
            f"roll_mean_{w}": shifted.rolling(w, min_periods=1).mean(),
            f"roll_std_{w}": shifted.rolling(w, min_periods=1).std().fillna(0),
            f"roll_min_{w}": shifted.rolling(w, min_periods=1).min(),
            f"roll_max_{w}": shifted.rolling(w, min_periods=1).max(),
        }
        for k, v in stats.items():
            df[k] = v
            roll_cols.append(k)

    hist_cols = [c for c in cfg.selected_historical_features if c in df.columns]
    future_known_cols = []
    if getattr(cfg, "use_future_known_covariates", False):
        future_known_cols = [c for c in cfg.selected_future_known_features if c in df.columns]
    for c in list(dict.fromkeys(hist_cols + future_known_cols)):
        df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill().fillna(0)

    time_cols = ["hour", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    seg_cols = [f"segment_{s}" for s in range(7)]
    context_cols = hist_cols + time_cols + seg_cols + roll_cols
    future_cols = time_cols + seg_cols + future_known_cols
    all_feature_cols = list(dict.fromkeys(context_cols + future_cols))
    df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    df = df.iloc[cfg.warmup_rows:].reset_index(drop=True)
    return df, context_cols, future_cols, hist_cols, future_known_cols


def build_samples(df, cfg, context_cols, future_cols):
    price = df["price_yj"].values.astype(np.float64)
    context = df[context_cols].values.astype(np.float64)
    future = df[future_cols].values.astype(np.float64)
    times = pd.to_datetime(df[cfg.time_col])
    times_np = df[cfg.time_col].values
    hours_all = times.dt.hour.values

    xm, xf, y, hours, segs, sample_times, bd, bw = [], [], [], [], [], [], [], []
    n = len(df)
    pred_len = cfg.pred_len

    # Target samples are complete calendar days starting at 00:00.
    # Each sample predicts the target day (24 hours).
    # Matrix rows: recent 3 complete days + same weekday 7/14/30 days ago = 6 x 24.
    for ts in range(30 * 24, n - pred_len + 1):
        if int(hours_all[ts]) != 0:
            continue
        te = ts + pred_len
        if te > n:
            continue
        if not np.all(hours_all[ts:te] == np.arange(24)):
            continue

        blocks = [
            price[ts-72:ts-48],
            price[ts-48:ts-24],
            price[ts-24:ts],
            price[ts-7*24:ts-7*24+24],
            price[ts-14*24:ts-14*24+24],
            price[ts-30*24:ts-30*24+24],
        ]
        if len(blocks) != cfg.rows or not all(len(b) == cfg.cols for b in blocks):
            continue
        block_hours = [
            hours_all[ts-72:ts-48],
            hours_all[ts-48:ts-24],
            hours_all[ts-24:ts],
            hours_all[ts-7*24:ts-7*24+24],
            hours_all[ts-14*24:ts-14*24+24],
            hours_all[ts-30*24:ts-30*24+24],
        ]
        if not all(np.all(h == np.arange(24)) for h in block_hours):
            continue

        yy = price[ts:te]
        fut = future[ts:te].reshape(-1)
        if len(yy) != pred_len or len(fut) != pred_len * len(future_cols):
            continue

        bday = price[ts-24:ts-24+pred_len]
        bweek = price[ts-7*24:ts-7*24+pred_len]
        if len(bday) != pred_len or len(bweek) != pred_len:
            continue

        hs = hours_all[ts:te].astype(np.int64)
        ss = np.array([segment_id(int(h)) for h in hs], dtype=np.int64)
        ctx_idx = max(ts - 1, 0)
        xm.append(np.stack(blocks, axis=0))
        xf.append(np.concatenate([context[ctx_idx], fut], axis=0))
        y.append(yy)
        hours.append(hs)
        segs.append(ss)
        sample_times.append(times_np[ts])
        bd.append(bday)
        bw.append(bweek)

    arrays = [np.asarray(a) for a in [xm, xf, y, hours, segs, sample_times, bd, bw]]
    order = np.argsort(arrays[5])
    return tuple(a[order].astype(np.float32) if a.dtype.kind in "fc" else a[order] for a in arrays)


def train_only_labels(y, train_size):
    yt = y[:train_size]
    hi = float(np.quantile(yt, 0.90))
    lo = float(np.quantile(yt, 0.10))
    slope = yt[:, -1] - yt[:, 0]
    up = float(np.quantile(slope, 0.67))
    down = float(np.quantile(slope, 0.33))
    vol_thr = float(np.quantile(yt.std(axis=1), 0.75))

    mx, mn = y.max(axis=1), y.min(axis=1)
    regime = np.zeros(len(y), dtype=np.int64)
    regime[mx >= hi] = 1
    regime[mn <= lo] = 2
    regime[(mx >= hi) & (mn <= lo)] = 1
    sl = y[:, -1] - y[:, 0]
    trend = np.ones(len(y), dtype=np.int64)
    trend[sl >= up] = 2
    trend[sl <= down] = 0
    vol = np.zeros(len(y), dtype=np.int64)
    vol[y.std(axis=1) >= vol_thr] = 1
    vol[(mx >= hi) | (mn <= lo)] = 2
    return (regime, trend, vol), {"high": hi, "low": lo, "up": up, "down": down, "vol": vol_thr}


class PriceMatrixDataset(Dataset):
    def __init__(self, matrix, feat, y, hours, segs, labels, bd, bw):
        self.matrix = torch.FloatTensor(matrix)
        self.feat = torch.FloatTensor(feat)
        self.y = torch.FloatTensor(y)
        self.hours = torch.LongTensor(hours)
        self.segs = torch.LongTensor(segs)
        self.regime = torch.LongTensor(labels[0])
        self.trend = torch.LongTensor(labels[1])
        self.vol = torch.LongTensor(labels[2])
        self.bd = torch.FloatTensor(bd)
        self.bw = torch.FloatTensor(bw)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.matrix[i], self.feat[i], self.y[i], self.hours[i], self.segs[i], self.regime[i], self.trend[i], self.vol[i], self.bd[i], self.bw[i]


def build_loaders(cfg, matrix, feat, y, hours, segs, labels, bd, bw):
    n = len(y)
    tr = int(n * cfg.train_ratio)
    va = int(n * cfg.val_ratio)
    scaler = StandardScaler()
    feat_scaled = feat.copy()
    feat_scaled[:tr] = scaler.fit_transform(feat[:tr])
    feat_scaled[tr:tr+va] = scaler.transform(feat[tr:tr+va])
    feat_scaled[tr+va:] = scaler.transform(feat[tr+va:])

    splits = [(0, tr), (tr, tr+va), (tr+va, n)]
    dsets = []
    for a, b in splits:
        labs = (labels[0][a:b], labels[1][a:b], labels[2][a:b])
        dsets.append(PriceMatrixDataset(matrix[a:b], feat_scaled[a:b], y[a:b], hours[a:b], segs[a:b], labs, bd[a:b], bw[a:b]))
    loaders = [
        DataLoader(dsets[0], cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers),
        DataLoader(dsets[1], cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers),
        DataLoader(dsets[2], cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers),
    ]
    return dsets, loaders, scaler


def normalize_market_columns(df, cfg):
    """
    GridPulse unified column mapping for Shandong/Liaoning.
    Keeps original columns but creates unified aliases when possible.
    """
    df=df.copy()
    alias = {
    "LOAD": ["Real-time load",  "Real-time tie-line load"],
    "WIND": ["wind output"],
    "SOLAR": ["Photovoltaic output"],
    "HYDRO": ["Hydro/pumped-storage output"],
    "RENEWABLE": ["renewable total"],
    "LINE": ["Real-time tie-line power"],
    "TEMP": ["2m air temperature"],
    "PRESSURE": ["Sea-level pressure", "surface pressure"],
    "HUMIDITY": ["2m relative humidity"],
    "WIND_SPEED": ["10m wind speed", "2m wind speed"],
    "Photovoltaic output": ["Photovoltaic output"],
    "wind output": ["wind output"],
    "Hydro/pumped-storage output": ["Hydro/pumped-storage output"],
    "Real-time load": ["Real-time load"],
    "Real-time tie-line power": ["Real-time tie-line power"]
}
    for new, cols in alias.items():
        if new in df.columns:
            continue
        for c in cols:
            if c in df.columns:
                df[new]=df[c]
                break
    return df
