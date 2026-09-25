"""Full-resolution internal validation for a selected lesion checkpoint."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image

from ml.artifact_integrity import sha256
from ml.datasets.segmentation import LESION_PATHS
from ml.models import build_lesion_model
from ml.lesions import gaussian_importance_map, retinal_field_mask
from ml.training.train_classifier_v2 import device_for_training
from ml.training.train_lesion_segmentation_v3 import select_thresholds

def starts(length,size,overlap):
    if length<=size:return [0]
    values=list(range(0,length-size+1,size-overlap))
    if values[-1]!=length-size:values.append(length-size)
    return values

def infer(model,rgb,size,overlap,device,sigma_scale=.125,border_margin_fraction=.015):
    h,w=rgb.shape[:2];ph=max(0,size-h);pw=max(0,size-w);work=np.pad(rgb,((0,ph),(0,pw),(0,0)),mode="reflect")
    total=np.zeros((4,*work.shape[:2]),np.float32);weight=np.zeros(work.shape[:2],np.float32);importance=gaussian_importance_map(size,sigma_scale)
    mean=np.asarray((.485,.456,.406),np.float32);std=np.asarray((.229,.224,.225),np.float32)
    with torch.no_grad():
        for top in starts(work.shape[0],size,overlap):
            for left in starts(work.shape[1],size,overlap):
                tile=((work[top:top+size,left:left+size]/255.-mean)/std).transpose(2,0,1)
                logits=model(torch.from_numpy(tile).float().unsqueeze(0).to(device))[0].detach().cpu().numpy()
                total[:,top:top+size,left:left+size]+=logits*importance;weight[top:top+size,left:left+size]+=importance
    probability=1/(1+np.exp(-np.clip(total/np.maximum(weight,1),-60,60)))
    probability=probability[:,:h,:w];field=retinal_field_mask(rgb,border_margin_fraction);probability[:,~field]=0
    return probability

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data-root",type=Path,default=Path(os.environ.get("RETINASATHI_DATA_ROOT","")));p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False);config=state["config"];classes=list(config["data"]["classes"]);prep=config["preprocessing"];grid=[float(x) for x in config["validation"]["threshold_grid"]]
    model=build_lesion_model(config,4,pretrained=False);model.load_state_dict(state["model"]);device=device_for_training();model=model.to(device).eval()
    with Path("runs/splits/idrid_segmentation_train_validation.csv").open() as handle:ids=[row["image_id"] for row in csv.DictReader(handle) if row["split"]=="validation"]
    base=args.data_root/"IDRiD/original_extracted/A. Segmentation";counts=np.zeros((len(grid),4,3),np.int64);per_image=[]
    for number,image_id in enumerate(ids,start=1):
        rgb=np.asarray(Image.open(base/f"1. Original Images/a. Training Set/{image_id}.jpg").convert("RGB"),np.float32)
        truth=[]
        for folder,suffix in LESION_PATHS.values():
            path=base/f"2. All Segmentation Groundtruths/a. Training Set/{folder}/{image_id}_{suffix}.tif"
            truth.append(np.asarray(Image.open(path).convert("L"))>0 if path.exists() else np.zeros(rgb.shape[:2],bool))
        truth=np.stack(truth);inference=config.get("inference",{});probability=infer(model,rgb,int(prep["patch_size"]),int(inference.get("overlap",prep["inference_overlap"])),device,float(inference.get("gaussian_sigma_scale",.125)),float(inference.get("retinal_border_margin_fraction",.015)))
        for threshold_index,threshold in enumerate(grid):
            predicted=probability>=threshold
            counts[threshold_index,:,0]+=(predicted&truth).sum((1,2));counts[threshold_index,:,1]+=(predicted&~truth).sum((1,2));counts[threshold_index,:,2]+=(~predicted&truth).sum((1,2))
        per_image.append({"image_id":image_id,"ground_truth_pixels":truth.sum((1,2)).astype(int).tolist()});print(f"[{number}/{len(ids)}] {image_id}",flush=True)
    thresholds,metrics=select_thresholds(counts,grid,classes);mean=float(np.mean([x["dice"] for x in metrics.values()]));minimum=float(config["validation"]["release_minimum_mean_dice"]);class_min={k:float(v) for k,v in config["validation"]["release_minimum_dice_per_class"].items()};class_pass={name:metrics[name]["dice"]>=class_min[name] for name in classes};passed=mean>=minimum and all(class_pass.values())
    report={"model_status":"experimental_candidate" if passed else "rejected_validation_gate","checkpoint_sha256":sha256(args.checkpoint),"class_order":classes,"validation":{"thresholds":thresholds,"per_class":metrics,"mean_dice":mean,"image_count":len(ids)},"inference":config.get("inference",{}),"release_gate":{"minimum_mean_dice":minimum,"minimum_dice_per_class":class_min,"class_pass":class_pass,"passed":passed},"validation_scope":"full-resolution Gaussian overlap-tile internal validation","images":per_image,"official_test_used":False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report["release_gate"]|{"mean_dice":mean}),flush=True)
if __name__=="__main__":main()
