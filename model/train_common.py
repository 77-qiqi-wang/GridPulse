# -*- coding: utf-8 -*-
import copy, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import ExtraTreesRegressor
from data import load_data_strict, add_causal_features, build_samples, train_only_labels, build_loaders, inverse_yj
from model import CDEARDiff


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def calc_metrics(y, p):
    y,p=np.asarray(y).reshape(-1),np.asarray(p).reshape(-1)
    mask=np.isfinite(y)&np.isfinite(p); y,p=y[mask],p[mask]
    rmse=float(np.sqrt(mean_squared_error(y,p)))
    denom=float(np.mean(np.abs(y)))
    rrmse=rmse/(denom+1e-8)*100
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":rmse,"RRMSE (%)":rrmse,"R2":float(r2_score(y,p)),"sMAPE (%)":float(np.mean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+1e-8))*100)}

def mad(x):
    x=np.asarray(x).reshape(-1); x=x[np.isfinite(x)]
    return float(np.median(np.abs(x-np.median(x)))) if len(x) else 0.0

def robust_scale(y):
    y=np.asarray(y).reshape(-1); y=y[np.isfinite(y)]
    s=1.4826*mad(np.diff(y)) if len(y)>2 else 1.0
    if not np.isfinite(s) or s<=1e-8: s=1.4826*mad(y)
    if not np.isfinite(s) or s<=1e-8: s=float(np.std(y))
    return max(s,1e-6)

def robust_metrics(y,p,scale):
    z=np.abs(p.reshape(-1)-y.reshape(-1))/(scale+1e-8); z=z[np.isfinite(z)]
    l=np.log1p(z); q25,q50,q75=np.quantile(l,[.25,.5,.75])
    return {"RSI":1/(1+float(np.median(l))+mad(l)),"QRS":1/(1+float(q50)+float(q75-q25)),"HARI":float(np.mean(1/(1+z)))}

class EMA:
    def __init__(self, model, decay): self.decay=decay; self.shadow={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}; self.backup={}
    def update(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad: self.shadow[n]=self.decay*self.shadow[n]+(1-self.decay)*p.detach()
    def apply(self, model):
        self.backup={}
        for n,p in model.named_parameters():
            if p.requires_grad: self.backup[n]=p.data.clone(); p.data=self.shadow[n].clone()
    def restore(self, model):
        for n,p in model.named_parameters():
            if p.requires_grad and n in self.backup: p.data=self.backup[n]

def class_weights(labels,n=3):
    c=np.bincount(labels,minlength=n).astype(np.float64); f=c/c.sum(); w=1/(f+0.1); return (w/w.sum()*n).astype(np.float32)

def regime_weight(reg):
    w=torch.ones_like(reg,dtype=torch.float32); w=torch.where(reg==1,torch.full_like(w,1.6),w); w=torch.where(reg==2,torch.full_like(w,1.25),w); return w

def sample_t(reg, steps, device):
    return torch.randint(0,steps,(len(reg),),device=device)

def residual_scale_from_train(y,bd,bw,cfg):
    res=np.stack([y-bd,y-bw],0)
    return np.maximum(np.quantile(np.abs(res),cfg.residual_quantile,axis=(0,1)).astype(np.float32),1e-3)

def run_epoch(model, loader, cfg, device, opt=None, sch=None, ema=None):
    train=opt is not None; model.train(train); total={k:0.0 for k in ["loss","anchor","final","robust","shape","v","x0","cls","gate"]}; seen=0
    for batch in loader:
        matrix,feat,y,hours,segs,reg,tr,vol,bd,bw=[x.to(device) for x in batch]
        t=sample_t(reg,cfg.diffusion_steps,device); noise=torch.randn_like(y)
        if train: opt.zero_grad(set_to_none=True)
        loss,losses,_,_,_,_,_=model.forward_train(matrix,feat,y,(reg,tr,vol),bd,bw,t,noise)
        if train:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); opt.step(); sch.step(); ema.update(model)
        b=len(y); seen+=b; total["loss"]+=float(loss.detach())*b
        for k,v in losses.items(): total[k]+=float(v.detach())*b
    return {k:v/max(seen,1) for k,v in total.items()}

@torch.no_grad()
def collect(model, loader, cfg, device, yj, cmin, cmax):
    model.eval(); ps,as_,bs,ys,regs,rps=[],[],[],[],[],[]
    for batch in loader:
        matrix,feat,y,hours,segs,reg,tr,vol,bd,bw=batch
        matrix,feat,bd,bw=matrix.to(device),feat.to(device),bd.to(device),bw.to(device)
        pred_s,anchor,base,weights,cls=model.predict(matrix,feat,bd,bw,cfg.mc_samples)
        pred=pred_s.median(0).values.cpu().numpy(); anchor=anchor.cpu().numpy(); base=base.cpu().numpy()
        ps.append(inverse_yj(yj,pred,cmin,cmax)); as_.append(inverse_yj(yj,anchor,cmin,cmax)); bs.append(inverse_yj(yj,base,cmin,cmax)); ys.append(inverse_yj(yj,y.numpy(),cmin,cmax)); regs.append(reg.numpy()); rps.append(cls[3].cpu().numpy())
    return np.concatenate(ps),np.concatenate(as_),np.concatenate(bs),np.concatenate(ys),np.concatenate(regs),np.concatenate(rps)

def apply_calibrator(pred, cal):
    if cal is None: return pred
    out=pred.copy()
    kind=cal.get("kind", "none")
    if kind=="hourly_offset":
        out=pred+cal["offset"].reshape(1,-1)
    elif kind=="hourly_affine":
        out=pred*cal["slope"].reshape(1,-1)+cal["intercept"].reshape(1,-1)
    elif kind=="hourly_mixed":
        off=pred+cal["offset"].reshape(1,-1)
        aff=pred*cal["slope"].reshape(1,-1)+cal["intercept"].reshape(1,-1)
        w=cal["mix"].reshape(1,-1)
        out=(1-w)*off+w*aff
    return np.clip(out, cal.get("clip_min", -1e9), cal.get("clip_max", 1e9))

def fit_post_calibrator(pred, y, cfg):
    if not getattr(cfg,"post_calibrate",False): return None
    h=pred.shape[1]; offsets=np.zeros(h,dtype=np.float32); slopes=np.ones(h,dtype=np.float32); intercepts=np.zeros(h,dtype=np.float32); mix=np.zeros(h,dtype=np.float32)
    clip_min=float(np.nanmin(y)-getattr(cfg,"post_calibrate_clip_margin",200.0)); clip_max=float(np.nanmax(y)+getattr(cfg,"post_calibrate_clip_margin",200.0))
    for i in range(h):
        x=pred[:,i].astype(np.float64); yy=y[:,i].astype(np.float64)
        raw_mae=np.mean(np.abs(x-yy))
        cand_offset=np.median(yy-x)
        off_mae=np.mean(np.abs(x+cand_offset-yy))
        if off_mae+1e-8 < raw_mae*float(getattr(cfg,"post_calibrate_offset_gain",0.999)):
            offsets[i]=cand_offset
        else:
            offsets[i]=0.0
        vx=np.var(x)
        if np.isfinite(vx) and vx>1e-8:
            cov=np.mean((x-x.mean())*(yy-yy.mean()))
            raw_s=np.clip(cov/(vx+float(getattr(cfg,"post_calibrate_ridge",10.0))),getattr(cfg,"post_calibrate_slope_min",0.65),getattr(cfg,"post_calibrate_slope_max",1.35))
            raw_b=yy.mean()-raw_s*x.mean()
            off=x+offsets[i]; aff=raw_s*x+raw_b
            mae_off=np.mean(np.abs(off-yy)); mae_aff=np.mean(np.abs(aff-yy)); mae_raw=np.mean(np.abs(x-yy))
            if mae_aff+1e-8 < min(mae_off,mae_raw)*float(getattr(cfg,"post_calibrate_affine_gain",0.995)):
                slopes[i]=raw_s; intercepts[i]=raw_b; mix[i]=float(getattr(cfg,"post_calibrate_affine_mix",0.55))
        if i+1 in getattr(cfg,"post_calibrate_hours_offset_only",()):
            slopes[i]=1.0; intercepts[i]=0.0; mix[i]=0.0
    return {"kind":"hourly_mixed","offset":offsets,"slope":slopes,"intercept":intercepts,"mix":mix,"clip_min":clip_min,"clip_max":clip_max}

def val_report(model, loader, cfg, device, yj, cmin, cmax):
    p,a,b,y,_,_=collect(model,loader,cfg,device,yj,cmin,cmax)
    rs=robust_scale(y.reshape(-1)); rb=robust_metrics(y,p,rs); cm=calc_metrics(y,p)
    score=cm["MAE"]-float(getattr(cfg,"val_metric_bonus",0.0))*(rb["RSI"]+rb["QRS"]+rb["HARI"])*rs
    return {"mae":cm["MAE"],"rmse":cm["RMSE"],"rrmse":cm["RRMSE (%)"],"r2":cm["R2"],"smape":cm["sMAPE (%)"],"anchor_mae":mean_absolute_error(y.reshape(-1),a.reshape(-1)),"base_mae":mean_absolute_error(y.reshape(-1),b.reshape(-1)),"score":score,**rb}

def fit_tree_auxiliary(cfg, feat, y, scaler, yj, cmin, cmax, tr, va, val_model_pred, outdir):
    if not getattr(cfg,"tree_auxiliary",False): return None, 0.0
    x=scaler.transform(feat)
    y_orig=inverse_yj(yj,y,cmin,cmax)
    val_slice=slice(tr,tr+va); test_slice=slice(tr+va,len(y))
    weights=np.asarray(getattr(cfg,"tree_aux_weight_grid",(0.0,0.25,0.5,0.75,1.0)),dtype=float)
    hourly=bool(getattr(cfg,"tree_aux_hourly",True))
    mid_hours=set(int(h) for h in getattr(cfg,"tree_aux_strong_hours",()))
    val_tree=np.zeros((va,cfg.pred_len),dtype=np.float32)
    test_tree=np.zeros((len(y)-tr-va,cfg.pred_len),dtype=np.float32)
    best_w=np.zeros(cfg.pred_len,dtype=np.float32)
    best_mae=np.zeros(cfg.pred_len,dtype=np.float32)
    if hourly:
        for h in range(cfg.pred_len):
            hh=h+1
            est=int(getattr(cfg,"tree_aux_estimators",160)); depth=int(getattr(cfg,"tree_aux_max_depth",8)); leaf=int(getattr(cfg,"tree_aux_min_samples_leaf",3))
            if hh in mid_hours:
                est=int(getattr(cfg,"tree_aux_mid_estimators",est)); depth=int(getattr(cfg,"tree_aux_mid_max_depth",depth)); leaf=int(getattr(cfg,"tree_aux_mid_min_samples_leaf",leaf))
            reg_val=ExtraTreesRegressor(n_estimators=est,max_depth=depth,min_samples_leaf=leaf,random_state=cfg.seed+hh,n_jobs=int(getattr(cfg,"tree_aux_n_jobs",1)))
            reg_val.fit(x[:tr],y_orig[:tr,h])
            val_tree[:,h]=reg_val.predict(x[val_slice])
            scores=[]
            for w in weights:
                blend=(1-w)*val_model_pred[:,h]+w*val_tree[:,h]
                scores.append(mean_absolute_error(y_orig[val_slice,h],blend))
            w=float(weights[int(np.argmin(scores))])
            w=float(np.clip(w,float(getattr(cfg,"tree_aux_min_weight",0.0)),float(getattr(cfg,"tree_aux_max_weight",1.0))))
            best_w[h]=w; best_mae[h]=min(scores)
            reg=ExtraTreesRegressor(n_estimators=est,max_depth=depth,min_samples_leaf=leaf,random_state=cfg.seed+100+hh,n_jobs=int(getattr(cfg,"tree_aux_n_jobs",1)))
            reg.fit(x[:tr+va],y_orig[:tr+va,h])
            test_tree[:,h]=reg.predict(x[test_slice])
    else:
        reg_val=ExtraTreesRegressor(n_estimators=int(getattr(cfg,"tree_aux_estimators",160)),max_depth=int(getattr(cfg,"tree_aux_max_depth",8)),min_samples_leaf=int(getattr(cfg,"tree_aux_min_samples_leaf",3)),random_state=cfg.seed,n_jobs=int(getattr(cfg,"tree_aux_n_jobs",1)))
        reg_val.fit(x[:tr],y_orig[:tr]); val_tree=reg_val.predict(x[val_slice])
        scores=[]
        for w in weights:
            blend=(1-w)*val_model_pred+w*val_tree
            scores.append(mean_absolute_error(y_orig[val_slice].reshape(-1),blend.reshape(-1)))
        w=float(weights[int(np.argmin(scores))]); w=float(np.clip(w,float(getattr(cfg,"tree_aux_min_weight",0.0)),float(getattr(cfg,"tree_aux_max_weight",1.0))))
        best_w[:]=w; best_mae[:]=min(scores)
        reg=ExtraTreesRegressor(n_estimators=int(getattr(cfg,"tree_aux_estimators",160)),max_depth=int(getattr(cfg,"tree_aux_max_depth",8)),min_samples_leaf=int(getattr(cfg,"tree_aux_min_samples_leaf",3)),random_state=cfg.seed,n_jobs=int(getattr(cfg,"tree_aux_n_jobs",1)))
        reg.fit(x[:tr+va],y_orig[:tr+va]); test_tree=reg.predict(x[test_slice])
    if hourly:
        pd.DataFrame({"H":np.arange(1,cfg.pred_len+1),"selected_weight":best_w,"val_mae":best_mae}).to_csv(outdir/"tree_aux_weight_search.csv",index=False,encoding="utf-8-sig")
        print(f"Tree auxiliary fitted: mean_weight={best_w.mean():.2f}, mean_hour_val_mae={best_mae.mean():.4f}")
        return test_tree, best_w
    pd.DataFrame({"weight_grid":weights,"val_mae":scores}).to_csv(outdir/"tree_aux_weight_search.csv",index=False,encoding="utf-8-sig")
    print(f"Tree auxiliary fitted: selected_weight={w:.2f}, val_mae={min(scores):.4f}")
    return test_tree, w

def evaluate(model, loader, cfg, device, yj, cmin, cmax, outdir, calibrator=None, tree_pred=None, tree_weight=0.0):
    p,a,b,y,reg,rp=collect(model,loader,cfg,device,yj,cmin,cmax)
    raw_p=p.copy()
    p=apply_calibrator(p, calibrator)
    if tree_pred is not None:
        tw=np.asarray(tree_weight,dtype=float)
        if tw.ndim==0: tw=float(tw)
        else: tw=tw.reshape(1,-1)
        p=(1.0-tw)*p+tw*tree_pred
    rs=robust_scale(y.reshape(-1))
    m=calc_metrics(y,p); m.update(robust_metrics(y,p,rs))
    raw_m=calc_metrics(y,raw_p); raw_m.update(robust_metrics(y,raw_p,rs))
    anchor_m=calc_metrics(y,a); anchor_m.update(robust_metrics(y,a,rs))
    base_m=calc_metrics(y,b); base_m.update(robust_metrics(y,b,rs))
    print("\nCDE-ARDiff Results")
    for k,v in m.items(): print(f"  {k}: {v:.4f}")
    print(f"  AnchorMAE: {anchor_m['MAE']:.4f}, BaseMAE: {base_m['MAE']:.4f}")
    rows=[]
    for h in range(cfg.pred_len):
        mh=calc_metrics(y[:,h],p[:,h]); mh.update(robust_metrics(y[:,h],p[:,h],rs)); rows.append({"H":h+1,**mh})
        print(f"H{h+1}: MAE={mh['MAE']:.2f}, RMSE={mh['RMSE']:.2f}, RRMSE={mh['RRMSE (%)']:.2f}%, R2={mh['R2']:.4f}, sMAPE={mh['sMAPE (%)']:.2f}%, RSI={mh['RSI']:.4f}, QRS={mh['QRS']:.4f}, HARI={mh['HARI']:.4f}")
    print("\nRegime classification")
    print(classification_report(reg,rp.argmax(1),labels=[0,1,2],target_names=["normal","high","low"],digits=4,zero_division=0))
    cols={}
    for h in range(cfg.pred_len):
        cols[f"True_H{h+1}"]=y[:,h]; cols[f"Pred_H{h+1}"]=p[:,h]; cols[f"RawPred_H{h+1}"]=raw_p[:,h]; cols[f"Anchor_H{h+1}"]=a[:,h]; cols[f"Base_H{h+1}"]=b[:,h]
        if tree_pred is not None: cols[f"TreeAux_H{h+1}"]=tree_pred[:,h]
    res=pd.DataFrame(cols)
    summary_rows=[{ "Model":"CDE-ARDiff", **m }, { "Model":"CDE-ARDiff-Raw", **raw_m }, { "Model":"Anchor", **anchor_m }, { "Model":"Base", **base_m }]
    if tree_pred is not None:
        tree_m=calc_metrics(y,tree_pred); tree_m.update(robust_metrics(y,tree_pred,rs)); summary_rows.append({"Model":"TreeAux", **tree_m})
    summary=pd.DataFrame(summary_rows)
    res.to_csv(outdir/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(outdir/"horizon_metrics.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(outdir/"metrics_summary.csv",index=False,encoding="utf-8-sig")

def main(cfg):
    cfg.ensure_dirs(); set_seed(cfg.seed); outdir=Path(cfg.output_dir); device=torch.device("cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu")
    print(f"Device: {device}\nCDE-ARDiff")
    df,yj,cmin,cmax=load_data_strict(cfg); df,ctx,fut,_,_=add_causal_features(df,cfg); matrix,feat,y,hours,segs,_,bd,bw=build_samples(df,cfg,ctx,fut)
    trn=int(len(y)*cfg.train_ratio); labels,th=train_only_labels(y,trn); dsets,loaders,scaler=build_loaders(cfg,matrix,feat,y,hours,segs,labels,bd,bw)
    print(f"Samples: train={len(dsets[0])}, val={len(dsets[1])}, test={len(dsets[2])}\nFeature dim: {dsets[0].feat.shape[1]}")
    cw=(class_weights(labels[0][:trn]),class_weights(labels[1][:trn]),class_weights(labels[2][:trn])); print(f"Auto class weights: reg={cw[0]}, tri={cw[1]}, vol={cw[2]}")
    model=CDEARDiff(cfg,dsets[0].feat.shape[1],residual_scale_from_train(y[:trn],bd[:trn],bw[:trn],cfg),cw).to(device); print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=cfg.lr,epochs=cfg.epochs,steps_per_epoch=len(loaders[0]),pct_start=0.12); ema=EMA(model,cfg.ema_decay)
    best,best_state,bad,logs=float("inf"),None,0,[]; t0=time.time()
    for ep in range(cfg.epochs):
        tr=run_epoch(model,loaders[0],cfg,device,opt,sch,ema); ema.apply(model); va=run_epoch(model,loaders[1],cfg,device); vr=val_report(model,loaders[1],cfg,device,yj,cmin,cmax); ema.restore(model)
        logs.append({"epoch":ep+1,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()},**{f"report_{k}":v for k,v in vr.items()}})
        print(f"Epoch [{ep+1:03d}/{cfg.epochs}] Train={tr['loss']:.4f} Val={va['loss']:.4f} MAE={vr['mae']:.3f} RMSE={vr['rmse']:.3f} RRMSE={vr['rrmse']:.2f}% R2={vr['r2']:.4f} sMAPE={vr['smape']:.2f}% Anchor={vr['anchor_mae']:.3f} Base={vr['base_mae']:.3f} RSI={vr['RSI']:.4f} QRS={vr['QRS']:.4f} HARI={vr['HARI']:.4f} Score={vr['score']:.3f}")
        if vr["score"]<best:
            best=vr["score"]; bad=0; ema.apply(model); best_state=copy.deepcopy(model.state_dict()); ema.restore(model); torch.save(best_state,outdir/"best_model.pth"); print("  * Best saved")
        else:
            bad+=1
            if bad>=cfg.patience: print("Early stopping"); break
    if best_state: model.load_state_dict(best_state)
    if int(getattr(cfg,"refit_train_val_epochs",0))>0:
        tv_loader=DataLoader(ConcatDataset([dsets[0],dsets[1]]), cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
        opt2=torch.optim.AdamW(model.parameters(),lr=float(getattr(cfg,"refit_lr",cfg.lr*0.25)),weight_decay=cfg.weight_decay)
        sch2=torch.optim.lr_scheduler.OneCycleLR(opt2,max_lr=float(getattr(cfg,"refit_lr",cfg.lr*0.25)),epochs=int(cfg.refit_train_val_epochs),steps_per_epoch=len(tv_loader),pct_start=0.20)
        ema2=EMA(model,cfg.ema_decay)
        print(f"Refit on train+val for {cfg.refit_train_val_epochs} epochs")
        for rp_ep in range(int(cfg.refit_train_val_epochs)):
            rr=run_epoch(model,tv_loader,cfg,device,opt2,sch2,ema2)
            print(f"  Refit [{rp_ep+1:03d}/{cfg.refit_train_val_epochs}] Loss={rr['loss']:.4f} Anchor={rr['anchor']:.4f} Final={rr['final']:.4f}")
        ema2.apply(model)
    pd.DataFrame(logs).to_csv(outdir/"training_log.csv",index=False,encoding="utf-8-sig"); print(f"Training done. best_val_score={best:.4f}, time={(time.time()-t0)/60:.1f}min")
    calibrator=None
    if getattr(cfg,"post_calibrate",False):
        vp,_,_,vy,_,_=collect(model,loaders[1],cfg,device,yj,cmin,cmax)
        calibrator=fit_post_calibrator(vp,vy,cfg)
        if calibrator is not None:
            pd.DataFrame({"H":np.arange(1,cfg.pred_len+1),"offset":calibrator["offset"],"slope":calibrator["slope"],"intercept":calibrator["intercept"],"mix":calibrator["mix"]}).to_csv(outdir/"post_calibrator.csv",index=False,encoding="utf-8-sig")
            print("Post calibrator fitted on validation set")
    tree_pred,tree_weight=None,0.0
    if getattr(cfg,"tree_auxiliary",False):
        vp,_,_,_,_,_=collect(model,loaders[1],cfg,device,yj,cmin,cmax)
        tree_pred,tree_weight=fit_tree_auxiliary(cfg,feat,y,scaler,yj,cmin,cmax,trn,int(len(y)*cfg.val_ratio),vp,outdir)
    evaluate(model,loaders[2],cfg,device,yj,cmin,cmax,outdir,calibrator,tree_pred,tree_weight); print(f"Outputs saved to: {outdir.resolve()}")
