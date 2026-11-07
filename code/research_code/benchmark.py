"""Controlled experimental scaffold for the family-balanced 38-class study.

No original MADBase/AHCD data, checkpoint or results are bundled. This script
runs only with independently prepared, verified, non-overlapping partitions.

Requirements: numpy, scipy, torch, scikit-learn.
Train: python benchmark.py train --train train.npz --val val.npz --config B0_flat_ce --out run_B0
Evaluate: python benchmark.py evaluate --test untouched_test.npz --checkpoint run_B0/best.pt --out result_B0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, Sampler
from models import CONFIGS, family_risk


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def load_partition(path: str):
    """Requires images scaled to [0,1] and aligned to 64x64 with declared provenance."""
    with np.load(path, allow_pickle=False) as obj:
        if "images" not in obj or "labels" not in obj:
            raise ValueError("Expected .npz keys 'images' and 'labels'")
        x = obj["images"].astype(np.float32)
        y = obj["labels"].astype(np.int64)
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4 or x.shape[1:] != (1, 64, 64):
        raise ValueError(f"Expected N x 1 x 64 x 64, got {x.shape}. Document resize and orientation beforehand.")
    if y.shape != (len(x),) or np.any(y < 0) or np.any(y >= 38):
        raise ValueError("Expected N integer labels in [0,37]")
    if not np.isfinite(x).all() or float(x.min()) < 0 or float(x.max()) > 1:
        raise ValueError("Images must be finite and scaled to [0,1]; do not normalize twice")
    if not np.any(y < 10) or not np.any(y >= 10):
        raise ValueError("Each partition must contain digits and letters")
    return x, y


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Fail instead of silently using a nondeterministic training operator.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def eval_metrics(y: np.ndarray, pred: np.ndarray, gate: np.ndarray | None = None):
    dig = y < 10
    let = ~dig
    # Compute each family F1 from the FULL 38-way confusion outcome.
    # Predictions made across the family boundary must count as false positives
    # for the predicted class and false negatives for the true class.
    f_d = f1_score(y, pred, labels=list(range(10)), average="macro", zero_division=0)
    f_l = f1_score(y, pred, labels=list(range(10,38)), average="macro", zero_division=0)
    out = {
        "accuracy_all": float(accuracy_score(y, pred)),
        "accuracy_digits": float(accuracy_score(y[dig], pred[dig])),
        "accuracy_letters": float(accuracy_score(y[let], pred[let])),
        "macro_f1_38": float(f1_score(y, pred, labels=list(range(38)), average="macro",zero_division=0)),
        "macro_f1_digits": float(f_d),
        "macro_f1_letters": float(f_l),
        "worst_family_macro_f1": float(min(f_d, f_l)),
        "family_confusion_from_final_prediction": confusion_matrix(let.astype(int), (pred>=10).astype(int), labels=[0,1]).tolist(),
        "confusion_38": confusion_matrix(y, pred, labels=list(range(38))).tolist(),
    }
    if gate is not None:
        out["family_gate_confusion"] = confusion_matrix(let.astype(int),gate,labels=[0,1]).tolist()
    return out


@torch.no_grad()
def get_predictions(model, x, y, device, batch=256, return_latency=False):
    model.eval()
    dl = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch)
    preds, gates = [], []
    forward_seconds=0.
    for xb, _ in dl:
        bx=xb.to(device)
        if device=='cuda':torch.cuda.synchronize()
        t0=time.perf_counter()
        logp, flogp = model(bx)
        if device=='cuda':torch.cuda.synchronize()
        forward_seconds+=time.perf_counter()-t0
        preds.append(logp.argmax(dim=1).cpu().numpy())
        gates.append(flogp.argmax(dim=1).cpu().numpy())
    if return_latency:
        return np.concatenate(preds), np.concatenate(gates), forward_seconds
    return np.concatenate(preds), np.concatenate(gates)



class FixedMixedFamilyBatchSampler(Sampler):
    """Identical label-family-mixed minibatch schedule for all study arms.

    Natural sample mix is retained. Rejection is only triggered when a random
    minibatch lacks an entire family, which is negligible at the planned
    batch size but must not create unequal training schedules across arms.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, seed: int):
        self.labels=np.asarray(labels);self.batch_size=batch_size;self.seed=seed;self.epoch=0
        self.full_batches=len(labels)//batch_size
        if self.full_batches<1 or min((labels<10).sum(),(labels>=10).sum())<self.full_batches:
            raise ValueError("Dataset too small to ensure mixed-family batches")

    def __len__(self): return self.full_batches

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        self.epoch+=1
        for attempt in range(100):
            order=rng.permutation(len(self.labels))[:self.full_batches*self.batch_size]
            parts=order.reshape(self.full_batches,self.batch_size)
            fam=(self.labels[parts]<10)
            if fam.any(axis=1).all() and (~fam).any(axis=1).all():
                yield from [list(row) for row in parts]
                return
        raise RuntimeError("Could not create family-mixed batches; revise sampler or batch size")

def run_train(args):
    if args.config not in CONFIGS:
        raise ValueError("Unknown config: " + args.config)
    seed_everything(args.seed)
    xtr, ytr = load_partition(args.train)
    xval, yval = load_partition(args.val)
    # Structural file check; record proper writer split separately in data provenance.
    if sha256(args.train) == sha256(args.val):
        raise ValueError("Identical training/validation file; cannot select an unbiased checkpoint")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cls, rule = CONFIGS[args.config]
    model = cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "initial.pt")
    trainset = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    mixed_sampler = FixedMixedFamilyBatchSampler(ytr,args.batch,args.seed)
    trainloader = DataLoader(trainset, batch_sampler=mixed_sampler, num_workers=0)
    history=[]; best=-1.; best_epoch=0
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; batches=0; incomplete=0
        for xb,yb in trainloader:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logp, _ = model(xb)
            loss, _ = family_risk(logp,yb,rule=rule,lam=args.lam,tau=args.tau)
            loss.backward();opt.step();total+=float(loss.detach());batches+=1
        if batches == 0: raise ValueError("No valid minibatches. Increase batch size.")
        pred, gate=get_predictions(model,xval,yval,device)
        met=eval_metrics(yval,pred,gate)
        row={"epoch":epoch,"train_batch_loss":total/batches,"skipped_mixed_family_batches":incomplete,**{k:v for k,v in met.items() if not k.endswith('confusion') and not k.startswith('confusion_')}}
        history.append(row)
        score=met["worst_family_macro_f1"]
        print(f"epoch {epoch:03d}: train loss {total/batches:.4f} | val worst-F1 {score:.4f} | digit {met['macro_f1_digits']:.4f} | letter {met['macro_f1_letters']:.4f}",flush=True)
        if score>best:
            best=score;best_epoch=epoch
            torch.save({"model":model.state_dict(),"config":args.config,"seed":args.seed,"epoch":epoch,
                        "train_sha256":sha256(args.train),"val_sha256":sha256(args.val),
                        "lambda":args.lam,"tau":args.tau,"validation_worst_f1":best},outdir/"best.pt")
    metadata={"config":args.config,"seed":args.seed,"epochs":args.epochs,"batch":args.batch,
              "learning_rate":args.lr,"train_file_sha256":sha256(args.train),"val_file_sha256":sha256(args.val),
              "lambda":args.lam,"tau":args.tau,"checkpoint_selection":"validation_worst_family_macro_f1",
              "best_epoch":best_epoch,"best_validation_worst_family_macro_f1":best,
              "train_family_counts":{"digits":int((ytr<10).sum()),"letters":int((ytr>=10).sum())},
              "val_family_counts":{"digits":int((yval<10).sum()),"letters":int((yval>=10).sum())},
              "model_parameters":sum(p.numel() for p in model.parameters()),
              "pytorch_version":torch.__version__,"device":device,"history":history}
    (outdir/"training_record.json").write_text(json.dumps(metadata,indent=2))


def get_corruption(x, kind, level, seed):
    """Deterministic test-only corruptions; same arrays used for all models."""
    if kind=="none": return x
    if kind=="noise":
        rng=np.random.default_rng(seed)
        return np.clip(x + rng.normal(0,level,size=x.shape).astype(np.float32),0,1)
    if kind=="blur":
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(x, sigma=(0,0,level,level)).astype(np.float32)
    if kind=="contrast":
        # Intensities are scaled around the gray mid-point; orientation is unchanged.
        return np.clip(0.5 + level*(x-0.5),0,1)
    raise ValueError(kind)


def run_evaluate(args):
    """TEST is loaded only by this command after checkpoint has been frozen."""
    seed_everything(args.corruption_seed)
    x, y=load_partition(args.test)
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if sha256(args.test) in {ckpt["train_sha256"],ckpt["val_sha256"]}:
        raise ValueError("TEST matches a training/validation file. Refusing leakage.")
    cls,rule=CONFIGS[ckpt["config"]]
    device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model=cls().to(device)
    model.load_state_dict(ckpt["model"])
    record={"config":ckpt["config"],"seed":ckpt["seed"],"chosen_epoch":ckpt["epoch"],
            "checkpoint_sha256":sha256(args.checkpoint),"inference_device":device,
            "training_sha256":ckpt["train_sha256"],"validation_sha256":ckpt["val_sha256"],
            "test_sha256":sha256(args.test),"corruption_seed":args.corruption_seed,
            "test_n":len(y),"corruption_tests":{}}
    tests=[("none",0.),("noise",0.05),("noise",0.10),("blur",0.8),("blur",1.3),
           ("contrast",0.75),("contrast",0.50)]
    sample_pred_arrays={"y_true":y.astype(np.int16)}
    with np.load(args.test,allow_pickle=False) as original_test:
        if "writer_groups" in original_test:
            sample_pred_arrays["writer_groups"]=original_test["writer_groups"]
    for kind,level in tests:
        started=time.perf_counter()
        perturbed=get_corruption(x,kind,level,args.corruption_seed)
        pred,gate,latency=get_predictions(model,perturbed,y,device,return_latency=True)
        result=eval_metrics(y,pred,gate)
        result["model_forward_seconds"] = latency
        result["forward_images_per_second"] = len(y)/latency if latency>0 else None
        result["total_evaluation_wall_seconds"] = time.perf_counter()-started
        key=f"{kind}_{level}"
        sample_pred_arrays[f"pred_{key}"]=pred.astype(np.int16)
        sample_pred_arrays[f"family_head_{key}"]=gate.astype(np.int8)
        record["corruption_tests"][f"{kind}_{level}"]=result
        print(f"{kind} {level}: worst-family macro-F1={result['worst_family_macro_f1']:.4f}",flush=True)
    Path(args.out).write_text(json.dumps(record,indent=2))
    if args.save_predictions:
        np.savez_compressed(args.save_predictions,**sample_pred_arrays)
        record["predictions_npz"] = str(args.save_predictions)
        record["predictions_sha256"] = sha256(args.save_predictions)
        Path(args.out).write_text(json.dumps(record,indent=2))


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    tr=sub.add_parser("train");tr.add_argument("--train",required=True);tr.add_argument("--val",required=True)
    tr.add_argument("--config",choices=sorted(CONFIGS),required=True);tr.add_argument("--out",required=True)
    tr.add_argument("--seed",type=int,default=7);tr.add_argument("--epochs",type=int,default=20)
    tr.add_argument("--batch",type=int,default=128);tr.add_argument("--lr",type=float,default=0.001)
    tr.add_argument("--lam",type=float,default=0.5);tr.add_argument("--tau",type=float,default=0.2)
    tr.add_argument("--cpu",action="store_true")
    te=sub.add_parser("evaluate");te.add_argument("--test",required=True)
    te.add_argument("--checkpoint",required=True);te.add_argument("--out",required=True)
    te.add_argument("--corruption-seed",type=int,default=1234);te.add_argument("--cpu",action="store_true")
    te.add_argument("--save-predictions",default=None,help="Save per-example predictions for paired statistical analysis")
    args=parser.parse_args()
    if args.command=="train":run_train(args)
    else:run_evaluate(args)

if __name__=="__main__":cli()
