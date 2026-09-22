"""Train genuine IDRiD lesion or DRIVE vessel segmentation models."""

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
from ml.datasets import SegmentationDataset
from ml.models import CompactUNet, dice_bce_loss
from ml.training.train_classifier_v2 import device_for_training, git_commit, hardware_report, seed_everything


def segmentation_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = torch.sigmoid(logits) >= 0.5; truth = targets >= 0.5; dims = (0, 2, 3)
    true_positive=(predicted & truth).sum(dim=dims);false_positive=(predicted & ~truth).sum(dim=dims);false_negative=(~predicted & truth).sum(dim=dims)
    return true_positive.cpu().numpy(),false_positive.cpu().numpy(),false_negative.cpu().numpy()


def metrics_from_counts(true_positive: np.ndarray, false_positive: np.ndarray, false_negative: np.ndarray) -> dict[str, object]:
    def values(numerator: np.ndarray, denominator: np.ndarray, undefined_when_no_truth: bool=False) -> list[float | None]:
        output=[]
        for index,(top,bottom) in enumerate(zip(numerator,denominator)):
            if undefined_when_no_truth and true_positive[index]+false_negative[index]==0:output.append(None)
            elif bottom==0:output.append(None)
            else:output.append(float(top/bottom))
        return output
    dice=values(2*true_positive,2*true_positive+false_positive+false_negative,True)
    iou=values(true_positive,true_positive+false_positive+false_negative,True)
    precision=values(true_positive,true_positive+false_positive)
    recall=values(true_positive,true_positive+false_negative,True)
    valid=[value for value in dice if value is not None]
    return {"dice_per_class":dice,"iou_per_class":iou,"precision_per_class":precision,"recall_per_class":recall,"mean_dice":float(np.mean(valid)) if valid else 0.0,"ground_truth_positive_pixels":(true_positive+false_negative).astype(int).tolist()}


def evaluate(model, loader, device):
    model.eval(); losses=[]; totals=None
    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device); logits=model(images); losses.append(float(dice_bce_loss(logits,targets).cpu()))
            counts=segmentation_counts(logits,targets);totals=[value.astype(np.int64) for value in counts] if totals is None else [left+right for left,right in zip(totals,counts)]
    if totals is None:raise RuntimeError("Empty segmentation validation loader")
    return {"loss":float(np.mean(losses)),**metrics_from_counts(*totals)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--task",choices=("lesions","vessels"),required=True)
    parser.add_argument("--config",type=Path); parser.add_argument("--data-root",type=Path); parser.add_argument("--resume",type=Path); parser.add_argument("--run-dir",type=Path)
    args=parser.parse_args(); config_path=args.config or Path("configs/lesions.yaml" if args.task=="lesions" else "configs/vessels.yaml")
    config=yaml.safe_load(config_path.read_text()); data_root=args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT",""))
    if not data_root.is_dir(): raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    seed=int(config["experiment"]["seed"]); seed_everything(seed); size=int(config["preprocessing"]["input_size"])
    manifest=Path("runs/splits/idrid_segmentation_train_validation.csv" if args.task=="lesions" else "runs/splits/drive_internal_train_validation.csv")
    channels=4 if args.task=="lesions" else 1; train=SegmentationDataset(data_root,manifest,"train",args.task,size,True); validation=SegmentationDataset(data_root,manifest,"validation",args.task,size,False)
    batch=int(config["training"]["batch_size"]);workers=int(config["training"].get("num_workers",0));loader_options={"num_workers":workers,"persistent_workers":workers>0,**({"prefetch_factor":2} if workers>0 else {})};train_loader=DataLoader(train,batch_size=batch,shuffle=True,**loader_options); validation_loader=DataLoader(validation,batch_size=batch,shuffle=False,**loader_options)
    device=device_for_training(); model=CompactUNet(channels).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    epochs=int(config["training"]["epochs"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs); start=0; best=-1.0
    if args.resume:
        state=torch.load(args.resume,map_location=device); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); start=state["epoch"]; best=state["best_mean_dice"]
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); run_dir=args.run_dir or Path("runs")/f"{stamp}_{args.task}_v1"; run_dir.mkdir(parents=True,exist_ok=bool(args.resume))
    (run_dir/"config.yaml").write_text(config_path.read_text()); (run_dir/"hardware.json").write_text(json.dumps(hardware_report(data_root),indent=2)+"\n")
    (run_dir/"provenance.json").write_text(json.dumps({"git_commit":git_commit(),"official_test_used":False,"task":args.task},indent=2)+"\n")
    history=[]; stale=0; accumulation=int(config["training"].get("gradient_accumulation",1))
    for epoch in range(start+1,epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,(images,targets,_) in enumerate(train_loader,start=1):
            images,targets=images.to(device),targets.to(device); loss=dice_bce_loss(model(images),targets)/accumulation; loss.backward(); losses.append(float(loss.detach().cpu())*accumulation)
            if step%accumulation==0 or step==len(train_loader): optimizer.step(); optimizer.zero_grad(set_to_none=True)
        scheduler.step(); metrics=evaluate(model,validation_loader,device); record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":metrics}; history.append(record)
        with (run_dir/"history.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        is_best=metrics["mean_dice"]>best
        if is_best:best=metrics["mean_dice"]
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_mean_dice":best,"config":config,"official_test_used":False}; torch.save(state,run_dir/"last.pt")
        if is_best:
            torch.save(state,run_dir/"best.pt"); (run_dir/"best_metrics.json").write_text(json.dumps(record,indent=2)+"\n"); stale=0
        else:
            stale+=1
            if stale>=6: break
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    best_path=run_dir/"best.pt"
    if best_path.exists():model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["model"])
    final=evaluate(model,validation_loader,device);class_names=config["data"].get("classes",["vessels"])
    (run_dir/"final_validation.json").write_text(json.dumps({"model_status":"experimental_segmentation","checkpoint_sha256":sha256(best_path),"task":args.task,"class_order":class_names,"validation":final,"official_test_used":False},indent=2)+"\n")


if __name__=="__main__": main()
