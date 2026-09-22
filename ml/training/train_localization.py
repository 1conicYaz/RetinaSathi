"""Train the IDRiD optic-disc/fovea localizer without official-test leakage."""

from __future__ import annotations

import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch,yaml
from torch.utils.data import DataLoader

from ml.artifact_integrity import sha256
from ml.datasets import IDRiDLocalizationDataset
from ml.models import DiscFoveaLocalizer,localization_loss
from ml.training.train_classifier_v2 import device_for_training,git_commit,hardware_report,seed_everything


def evaluate(model,loader,device):
    model.eval(); errors=[]; uncertainties=[]; losses=[]
    with torch.no_grad():
        for images,target,_ in loader:
            predicted,log_variance=model(images.to(device)); target=target.to(device); losses.append(float(localization_loss(predicted,log_variance,target).cpu()))
            difference=(predicted-target).reshape(-1,2,2); errors.append(torch.linalg.vector_norm(difference,dim=2).cpu().numpy())
            uncertainties.append(torch.exp(0.5*log_variance).reshape(-1,2,2).mean(dim=2).cpu().numpy())
    values=np.concatenate(errors)
    uncertainty=np.concatenate(uncertainties)
    return {"loss":float(np.mean(losses)),"disc_mean_normalized_error":float(values[:,0].mean()),"fovea_mean_normalized_error":float(values[:,1].mean()),"disc_p95_normalized_error":float(np.percentile(values[:,0],95)),"fovea_p95_normalized_error":float(np.percentile(values[:,1],95)),"disc_mean_predicted_std":float(uncertainty[:,0].mean()),"fovea_mean_predicted_std":float(uncertainty[:,1].mean())}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,default=Path("configs/localization.yaml"));parser.add_argument("--data-root",type=Path);parser.add_argument("--resume",type=Path);parser.add_argument("--run-dir",type=Path);args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text());data_root=args.data_root or Path(os.environ.get("RETINASATHI_DATA_ROOT",""))
    if not data_root.is_dir():raise SystemExit("Set RETINASATHI_DATA_ROOT or pass --data-root")
    seed_everything(int(config["experiment"]["seed"]));size=int(config["preprocessing"]["input_size"]);manifest=Path("runs/splits/idrid_internal_train_validation.csv")
    train=IDRiDLocalizationDataset(data_root,manifest,"train",size,True);validation=IDRiDLocalizationDataset(data_root,manifest,"validation",size,False);batch=int(config["training"]["batch_size"])
    workers=int(config["training"].get("num_workers",0));loader_options={"num_workers":workers,"persistent_workers":workers>0,**({"prefetch_factor":2} if workers>0 else {})};train_loader=DataLoader(train,batch_size=batch,shuffle=True,**loader_options);validation_loader=DataLoader(validation,batch_size=batch,shuffle=False,**loader_options);device=device_for_training();model=DiscFoveaLocalizer(pretrained=args.resume is None).to(device)
    epochs=int(config["training"]["epochs"]);optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs);start=0;best=float("inf")
    if args.resume:
        state=torch.load(args.resume,map_location=device);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);start=state["epoch"];best=state["best_loss"]
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ");run_dir=args.run_dir or Path("runs")/f"{stamp}_localization_v1";run_dir.mkdir(parents=True,exist_ok=bool(args.resume));(run_dir/"config.yaml").write_text(args.config.read_text());(run_dir/"hardware.json").write_text(json.dumps(hardware_report(data_root),indent=2)+"\n");(run_dir/"provenance.json").write_text(json.dumps({"git_commit":git_commit(),"official_test_used":False},indent=2)+"\n")
    stale=0;history=[]
    for epoch in range(start+1,epochs+1):
        model.train();losses=[]
        for images,target,_ in train_loader:
            optimizer.zero_grad(set_to_none=True);coordinates,log_variance=model(images.to(device));loss=localization_loss(coordinates,log_variance,target.to(device));loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        scheduler.step();metrics=evaluate(model,validation_loader,device);record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":metrics};history.append(record)
        with (run_dir/"history.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(record)+"\n")
        print(json.dumps(record),flush=True)
        is_best=metrics["loss"]<best
        if is_best:best=metrics["loss"]
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"best_loss":best,"config":config,"official_test_used":False};torch.save(state,run_dir/"last.pt")
        if is_best:torch.save(state,run_dir/"best.pt");(run_dir/"best_metrics.json").write_text(json.dumps(record,indent=2)+"\n");stale=0
        else:
            stale+=1
            if stale>=int(config["training"]["early_stopping_patience"]):break
    (run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    best_path=run_dir/"best.pt"
    if best_path.exists():model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["model"])
    (run_dir/"final_validation.json").write_text(json.dumps({"model_status":"experimental_localization","checkpoint_sha256":sha256(best_path),"coordinate_order":["optic_disc_x","optic_disc_y","fovea_x","fovea_y"],"confidence_definition":"relative confidence derived from learned heteroscedastic coordinate uncertainty; not a correctness probability","validation":evaluate(model,validation_loader,device),"official_test_used":False},indent=2)+"\n")


if __name__=="__main__":main()
