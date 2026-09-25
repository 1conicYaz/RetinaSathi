"""Train RetinaSathi's experimental IDRiD lesion segmentation candidate."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ml.artifact_integrity import sha256
from ml.datasets import IDRiDLesionPatchDataset
from ml.models import build_lesion_model, focal_tversky_loss
from ml.training.train_classifier_v2 import device_for_training, git_commit, hardware_report, seed_everything


def counts_for_thresholds(logits: torch.Tensor, targets: torch.Tensor, thresholds: list[float]) -> np.ndarray:
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    truth = targets.detach().cpu().numpy() >= .5
    counts = np.zeros((len(thresholds), probabilities.shape[1], 3), dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        predicted = probabilities >= threshold
        counts[index, :, 0] = (predicted & truth).sum(axis=(0,2,3))
        counts[index, :, 1] = (predicted & ~truth).sum(axis=(0,2,3))
        counts[index, :, 2] = (~predicted & truth).sum(axis=(0,2,3))
    return counts


def _safe_ratio(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return np.divide(top, bottom, out=np.zeros_like(top, dtype=np.float64), where=bottom > 0)


def select_thresholds(counts: np.ndarray, threshold_grid: list[float], classes: list[str]) -> tuple[dict[str,float], dict[str,dict[str,float]]]:
    dice = _safe_ratio(2*counts[:,:,0], 2*counts[:,:,0]+counts[:,:,1]+counts[:,:,2])
    selected: dict[str,float] = {}; metrics: dict[str,dict[str,float]] = {}
    for channel, name in enumerate(classes):
        best = int(np.argmax(dice[:,channel])); tp, fp, fn = counts[best,channel]
        selected[name] = float(threshold_grid[best])
        metrics[name] = {
            "dice": float(dice[best,channel]),
            "iou": float(_safe_ratio(np.asarray(tp), np.asarray(tp+fp+fn))),
            "precision": float(_safe_ratio(np.asarray(tp), np.asarray(tp+fp))),
            "recall": float(_safe_ratio(np.asarray(tp), np.asarray(tp+fn))),
            "ground_truth_positive_pixels": int(tp+fn),
        }
    return selected, metrics


def evaluate(model, loader, device, threshold_grid, classes, max_batches: int | None = None):
    model.eval(); total_counts=None; losses=[]
    with torch.no_grad():
        for batch_index,(images, targets, _) in enumerate(loader,start=1):
            images, targets = images.to(device), targets.to(device)
            logits = model(images); losses.append(float(focal_tversky_loss(logits,targets).cpu()))
            batch_counts = counts_for_thresholds(logits,targets,threshold_grid)
            total_counts = batch_counts if total_counts is None else total_counts + batch_counts
            if max_batches is not None and batch_index>=max_batches: break
    if total_counts is None: raise RuntimeError("Empty lesion validation loader")
    thresholds, per_class = select_thresholds(total_counts,threshold_grid,classes)
    return {"loss":float(np.mean(losses)),"thresholds":thresholds,"per_class":per_class,"mean_dice":float(np.mean([v["dice"] for v in per_class.values()]))}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/lesions_v3.yaml"))
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--epochs",type=int,help="Override epochs for a smoke run")
    parser.add_argument("--max-train-batches",type=int,help="Bound work for implementation smoke tests")
    parser.add_argument("--max-validation-batches",type=int,help="Bound work for implementation smoke tests")
    parser.add_argument("--initial-checkpoint",type=Path,help="Fine-tune from an existing lesion checkpoint without reusing optimizer state")
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()); data_root=args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT",""))
    if not data_root.is_dir(): raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    seed=int(config["experiment"]["seed"]); seed_everything(seed); device=device_for_training()
    prep=config["preprocessing"]; training=config["training"]; classes=list(config["data"]["classes"])
    manifest=Path("runs/splits/idrid_segmentation_train_validation.csv")
    hard_negative_path=Path(config["data"]["hard_negative_manifest"]) if config["data"].get("hard_negative_manifest") else None
    train=IDRiDLesionPatchDataset(data_root,manifest,"train",int(prep["patch_size"]),int(prep["training_samples_per_image"]),True,seed,hard_negative_path)
    validation=IDRiDLesionPatchDataset(data_root,manifest,"validation",int(prep["patch_size"]),int(prep["validation_samples_per_image"]),False,seed)
    options={"batch_size":int(training["batch_size"]),"num_workers":int(training.get("num_workers",0)),"pin_memory":device.type=="cuda"}
    train_loader=DataLoader(train,shuffle=True,**options); validation_loader=DataLoader(validation,shuffle=False,**options)
    model=build_lesion_model(config,len(classes),pretrained=True).to(device)
    if args.initial_checkpoint:
        initial=torch.load(args.initial_checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(initial["model"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(training["learning_rate"]),weight_decay=float(training["weight_decay"]))
    epochs=args.epochs or int(training["epochs"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    run_dir=args.run_dir or Path("runs")/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_lesions_v3")
    run_dir.mkdir(parents=True,exist_ok=False); (run_dir/"config.yaml").write_text(args.config.read_text())
    (run_dir/"hardware.json").write_text(json.dumps(hardware_report(data_root),indent=2)+"\n")
    (run_dir/"provenance.json").write_text(json.dumps({"git_commit":git_commit(),"official_test_used":False,"split_manifest":str(manifest),"initial_checkpoint":str(args.initial_checkpoint) if args.initial_checkpoint else None,"initial_checkpoint_sha256":sha256(args.initial_checkpoint) if args.initial_checkpoint else None,"validation_scope":"internal lesion-centred and hard-negative retinal patches"},indent=2)+"\n")
    best=-1.; stale=0; accumulation=int(training["gradient_accumulation"]); grid=[float(x) for x in config["validation"]["threshold_grid"]]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,(images,targets,_) in enumerate(train_loader,start=1):
            images,targets=images.to(device),targets.to(device)
            loss=focal_tversky_loss(model(images),targets)/accumulation; loss.backward(); losses.append(float(loss.detach().cpu())*accumulation)
            if step%accumulation==0 or step==len(train_loader): optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step==1 or step%10==0:
                (run_dir/"progress.json").write_text(json.dumps({"phase":"training","epoch":epoch,"batch":step,"batches":len(train_loader),"running_loss":float(np.mean(losses))})+"\n")
            if args.max_train_batches is not None and step>=args.max_train_batches:
                if step%accumulation!=0: optimizer.step(); optimizer.zero_grad(set_to_none=True)
                break
        (run_dir/"progress.json").write_text(json.dumps({"phase":"validation","epoch":epoch,"batch":0,"batches":len(validation_loader)})+"\n")
        scheduler.step(); validation_result=evaluate(model,validation_loader,device,grid,classes,args.max_validation_batches)
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"learning_rate":optimizer.param_groups[0]["lr"],"validation":validation_result}
        with (run_dir/"history.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        improved=validation_result["mean_dice"]>best
        if improved: best=validation_result["mean_dice"]; stale=0
        else: stale+=1
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"best_mean_dice":best,"config":config,"official_test_used":False}
        torch.save(state,run_dir/"last.pt")
        if improved:
            torch.save(state,run_dir/"best.pt"); (run_dir/"best_metrics.json").write_text(json.dumps(record,indent=2)+"\n")
        if stale>=int(training["early_stopping_patience"]): break
    best_path=run_dir/"best.pt"; state=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(state["model"])
    final=evaluate(model,validation_loader,device,grid,classes,args.max_validation_batches); gate=float(config["validation"]["release_minimum_mean_dice"])
    class_gates={name:float(value) for name,value in config["validation"]["release_minimum_dice_per_class"].items()}
    class_pass={name:final["per_class"][name]["dice"]>=class_gates[name] for name in classes}
    smoke=bool(args.max_train_batches or args.max_validation_batches);passed=(not smoke and final["mean_dice"]>=gate and all(class_pass.values()))
    report={"model_status":"smoke_test_only" if smoke else "training_candidate" if passed else "rejected_training_gate","checkpoint_sha256":sha256(best_path),"class_order":classes,"validation":final,"training_selection_gate":{"minimum_mean_dice":gate,"minimum_dice_per_class":class_gates,"class_pass":class_pass,"passed":passed},"release_gate":{"passed":False,"reason":"Full-resolution overlap-tile validation has not run"},"validation_scope":"implementation smoke test" if smoke else "internal lesion-centred and retinal-background patches","official_test_used":False}
    (run_dir/"final_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"run_dir":str(run_dir),**report["release_gate"],"mean_dice":final["mean_dice"]}),flush=True)


if __name__=="__main__": main()
