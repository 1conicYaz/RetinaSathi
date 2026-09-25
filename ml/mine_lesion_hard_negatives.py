"""Mine high-confidence false lesion candidates from the frozen training split."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation
from ml.datasets.segmentation import LESION_PATHS
from ml.evaluate_lesion_full_images import infer
from ml.models import build_lesion_model
from ml.training.train_classifier_v2 import device_for_training

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--data-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--per-image",type=int,default=12);args=parser.parse_args()
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False);config=state["config"];classes=list(config["data"]["classes"]);device=device_for_training();model=build_lesion_model(config,len(classes),pretrained=False).to(device).eval();model.load_state_dict(state["model"])
    with Path("runs/splits/idrid_segmentation_train_validation.csv").open() as handle: ids=[row["image_id"] for row in csv.DictReader(handle) if row["split"]=="train"]
    base=args.data_root/"IDRiD/original_extracted/A. Segmentation";size=int(config["preprocessing"]["patch_size"]);settings=config.get("inference",{});rows=[]
    for number,image_id in enumerate(ids,1):
        rgb=np.asarray(Image.open(base/f"1. Original Images/a. Training Set/{image_id}.jpg").convert("RGB"),np.float32);truth=[]
        for folder,suffix in LESION_PATHS.values():
            path=base/f"2. All Segmentation Groundtruths/a. Training Set/{folder}/{image_id}_{suffix}.tif";truth.append(np.asarray(Image.open(path).convert("L"))>0 if path.exists() else np.zeros(rgb.shape[:2],bool))
        exclusion=binary_dilation(np.any(np.stack(truth),axis=0),iterations=12);probability=infer(model,rgb,size,int(settings.get("overlap",128)),device,float(settings.get("gaussian_sigma_scale",.125)),float(settings.get("retinal_border_margin_fraction",.015)));candidate=probability.copy();candidate[:,exclusion]=0
        flat=candidate.reshape(len(classes),-1);chosen=[]
        for channel,name in enumerate(classes):
            order=np.argpartition(flat[channel],-min(flat.shape[1],args.per_image*200))[-args.per_image*200:];order=order[np.argsort(flat[channel,order])[::-1]]
            points=[]
            for index in order:
                y,x=divmod(int(index),rgb.shape[1]);score=float(flat[channel,index])
                if score<.25:break
                if all((x-px)**2+(y-py)**2 >= (size//3)**2 for px,py in points):points.append((x,y));chosen.append({"image_id":image_id,"class":name,"x":x,"y":y,"score":round(score,6)})
                if len(points)>=args.per_image:break
        rows.extend(chosen);print(f"[{number}/{len(ids)}] {image_id}: {len(chosen)}",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["image_id","class","x","y","score"]);writer.writeheader();writer.writerows(rows)
    print(f"saved {len(rows)} hard negatives to {args.output}",flush=True)
if __name__=="__main__":main()
