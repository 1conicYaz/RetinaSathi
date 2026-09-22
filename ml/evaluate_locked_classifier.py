"""One-way official IDRiD test and compatible Messidor external evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from ml.evaluation import classifier_metrics
from ml.artifact_integrity import sha256
from ml.training.train_classifier_v2 import RetinalTransform, build_model, device_for_training, grade_probabilities


class EvaluationDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], image_paths: dict[str, Path], transform) -> None:
        self.rows, self.image_paths, self.transform = rows, image_paths, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]; name = str(row["image_name"])
        with Image.open(self.image_paths[name]) as source:
            image = source.convert("RGB")
        return self.transform(image), int(row["grade"]), int(row.get("dme", -1))


def infer(model, loader, device, objective: str, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); grades=[]; probabilities=[]; dme_truth=[]; dme_probabilities=[]
    with torch.inference_mode():
        for images, truth, dme in loader:
            grade_logits, dme_logits = model(images.to(device))
            probabilities.append(grade_probabilities(grade_logits.cpu(), objective, temperature).numpy()); grades.append(truth.numpy())
            if dme_logits is not None:dme_probabilities.append(torch.softmax(dme_logits.cpu(),1).numpy());dme_truth.append(dme.numpy())
    return np.concatenate(grades),np.concatenate(probabilities),np.concatenate(dme_truth) if dme_truth else np.asarray([]),np.concatenate(dme_probabilities) if dme_probabilities else np.asarray([])


def bootstrap_binary(truth: np.ndarray, predicted: np.ndarray, seed: int = 26038, samples: int = 2000) -> dict[str, list[float]]:
    rng=np.random.default_rng(seed); sensitivity=[];specificity=[];balanced=[]
    for _ in range(samples):
        indices=rng.integers(0,len(truth),len(truth));actual=truth[indices];estimate=predicted[indices]
        sensitivity.append(float(((estimate==1)&(actual==1)).sum()/max((actual==1).sum(),1)));specificity.append(float(((estimate==0)&(actual==0)).sum()/max((actual==0).sum(),1)));balanced.append((sensitivity[-1]+specificity[-1])/2)
    return {"sensitivity_95_ci":np.percentile(sensitivity,[2.5,97.5]).tolist(),"specificity_95_ci":np.percentile(specificity,[2.5,97.5]).tolist(),"balanced_accuracy_95_ci":np.percentile(balanced,[2.5,97.5]).tolist()}


def idrid_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, Path]]:
    labels=root/"IDRiD/grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv";images=root/"IDRiD/images/B. Disease Grading/1. Original Images/b. Testing Set"
    with labels.open(encoding="utf-8-sig",newline="") as handle:
        rows=[{"image_name":row["Image name"],"grade":int(row["Retinopathy grade"]),"dme":int(row["Risk of macular edema "])} for row in csv.DictReader(handle)]
    return rows,{str(row["image_name"]):images/f"{row['image_name']}.jpg" for row in rows}


def messidor_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, Path]]:
    frames=[pd.read_excel(path) for path in sorted((root/"Messidor/annotations").rglob("*.xls"))];table=pd.concat(frames,ignore_index=True);table.columns=[str(column).strip() for column in table.columns]
    rows=[{"image_name":str(row["Image name"]),"grade":int(row["Retinopathy grade"]),"dme":int(row["Risk of macular edema"])} for _,row in table.iterrows()]
    paths={path.name:path for path in (root/"Messidor/images").rglob("*.tif")};return rows,paths


def binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    predicted=score>=threshold;sensitivity=float((predicted&truth).sum()/max(truth.sum(),1));specificity=float(((~predicted)&(~truth)).sum()/max((~truth).sum(),1))
    return {"sensitivity":sensitivity,"specificity":specificity,"balanced_accuracy":float((sensitivity+specificity)/2),"confusion_matrix":confusion_matrix(truth,predicted,labels=[False,True]).tolist(),**bootstrap_binary(truth.astype(int),predicted.astype(int))}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--validation",type=Path,required=True);parser.add_argument("--lock",type=Path,required=True);parser.add_argument("--data-root",type=Path,required=True);parser.add_argument("--output",type=Path,default=Path("artifacts/locked_evaluation.json"));args=parser.parse_args()
    if args.output.exists():raise SystemExit(f"Evaluation already exists: {args.output}. The official test is one-way; do not overwrite it.")
    lock=json.loads(args.lock.read_text())
    if lock.get("status")!="locked_pending_test" or sha256(args.checkpoint)!=lock.get("checkpoint_sha256") or sha256(args.validation)!=lock.get("validation_sha256"):raise SystemExit("Checkpoint/validation does not match the immutable pre-test lock")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False);config=state["config"];objective=str(config["training"]["objective"]);temperature=float(lock["temperature"]);threshold=float(lock["referable_threshold"]);transform=RetinalTransform(config,False);device=device_for_training();model=build_model(config,pretrained=False);model.load_state_dict(state["model"]);model.to(device)
    id_rows,id_paths=idrid_rows(args.data_root);id_truth,id_prob,id_dme,id_dme_prob=infer(model,DataLoader(EvaluationDataset(id_rows,id_paths,transform),batch_size=8,shuffle=False,num_workers=0),device,objective,temperature)
    id_metrics=classifier_metrics(id_truth,id_prob,threshold)
    if id_dme.size:
        id_metrics["dme_accuracy"]=float(accuracy_score(id_dme,id_dme_prob.argmax(1)));id_metrics["dme_macro_f1"]=float(f1_score(id_dme,id_dme_prob.argmax(1),average="macro",zero_division=0))
    else:
        id_metrics["dme_accuracy"]=None;id_metrics["dme_macro_f1"]=None
    me_rows,me_paths=messidor_rows(args.data_root);me_truth,me_prob,me_dme,me_dme_prob=infer(model,DataLoader(EvaluationDataset(me_rows,me_paths,transform),batch_size=8,shuffle=False,num_workers=0),device,objective,temperature);external=binary_metrics(me_truth>=2,me_prob[:,2:].sum(1),threshold)
    external_dme={"accuracy":float(accuracy_score(me_dme,me_dme_prob.argmax(1))),"macro_f1":float(f1_score(me_dme,me_dme_prob.argmax(1),average="macro",zero_division=0))} if me_dme.size else None
    result={"status":"locked_test_and_external_evaluation_complete","checkpoint_sha256":lock["checkpoint_sha256"],"validation_sha256":lock["validation_sha256"],"selection_sha256":lock["selection_sha256"],"lock_sha256":sha256(args.lock),"official_test_used_after_lock":True,"temperature_from_internal_validation":temperature,"referable_threshold_from_internal_validation":threshold,"idrid_official_test":{"samples":len(id_truth),"metrics":id_metrics},"messidor_external":{"samples":len(me_truth),"endpoint":"grade >=2 referable DR","metrics":external,"dme_risk_0_to_2":external_dme,"limitations":["Four-grade external scale evaluated only as a compatible binary endpoint","DME risk is reported separately only when the locked model has a DME head; label definitions may differ","Different centers, cameras, and population introduce domain shift","No Messidor result influenced selection, calibration, or threshold"]}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(args.output),"idrid_samples":len(id_truth),"messidor_samples":len(me_truth)}))


if __name__=="__main__":main()
