"""Export an experimental lesion or vessel model with its validation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from ml.artifact_integrity import require_internal_validation_binding, sha256
from ml.models import build_lesion_model


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--validation",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--version",required=True);args=parser.parse_args()
    manifest_path=args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():raise SystemExit(f"Refusing to overwrite export: {args.output}")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False);validation=json.loads(args.validation.read_text())
    try:checkpoint_digest=require_internal_validation_binding(args.checkpoint,validation)
    except ValueError as error:raise SystemExit(f"Refusing export: {error}") from error
    config=state["config"];task=validation.get("task","lesions");classes=validation["class_order"]
    if validation.get("release_gate",{}).get("passed") is False:
        raise SystemExit("Refusing export: lesion candidate failed its validation gate")
    if validation.get("validation_scope") not in {"full-resolution overlap-tile internal validation","full-resolution Gaussian overlap-tile internal validation"}:
        raise SystemExit("Refusing export: lesion candidate lacks full-resolution overlap-tile validation")
    channels=4 if task=="lesions" else 1;model=build_lesion_model(config,channels,pretrained=False);model.load_state_dict(state["model"]);model.eval();size=int(config["preprocessing"].get("patch_size",config["preprocessing"].get("input_size",512)));args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,3,size,size),args.output,input_names=["image"],output_names=["mask_logits"],opset_version=17,dynamo=False)
    inference=config.get("inference",{});onnx.checker.check_model(onnx.load(args.output));manifest={"schema_version":"2.0","model_version":args.version,"status":"ready_experimental","task":task,"dataset":config["data"]["dataset"],"classes":classes,"input_size":size,"thresholds":validation["validation"].get("thresholds",{name:.5 for name in classes}),"inference":{"mode":"gaussian_overlap_tiles","tile_size":size,"overlap":int(inference.get("overlap",config["preprocessing"].get("inference_overlap",128))),"gaussian_sigma_scale":float(inference.get("gaussian_sigma_scale",.125)),"retinal_border_margin_fraction":float(inference.get("retinal_border_margin_fraction",.015)),"minimum_region_pixels":int(config.get("validation",{}).get("minimum_region_pixels",3))},"preprocessing":{"retinal_crop":False,"aspect_ratio_safe_resize":False,"enhancement":"none","normalization":{"mean":[.485,.456,.406],"std":[.229,.224,.225]},"resampling":"native-resolution tiles"},"validation":validation["validation"],"validation_scope":validation.get("validation_scope"),"release_gate":validation.get("release_gate"),"official_test_used":False,"checkpoint_sha256":checkpoint_digest,"validation_sha256":sha256(args.validation),"onnx_sha256":sha256(args.output),"checkpoint_epoch":state["epoch"]}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"output":str(args.output),"task":task,"sha256":manifest["onnx_sha256"]}))


if __name__=="__main__":main()
