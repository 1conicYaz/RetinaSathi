"""Export the experimental optic-disc/fovea localizer and uncertainty contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from ml.artifact_integrity import require_internal_validation_binding, sha256
from ml.models import DiscFoveaLocalizer


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--validation",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--version",required=True);args=parser.parse_args()
    manifest_path=args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():raise SystemExit(f"Refusing to overwrite export: {args.output}")
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False);validation=json.loads(args.validation.read_text())
    try:checkpoint_digest=require_internal_validation_binding(args.checkpoint,validation)
    except ValueError as error:raise SystemExit(f"Refusing export: {error}") from error
    model=DiscFoveaLocalizer(pretrained=False);model.load_state_dict(state["model"]);model.eval();size=int(state["config"]["preprocessing"]["input_size"]);args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,torch.zeros(1,3,size,size),args.output,input_names=["image"],output_names=["coordinates","log_variance"],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(args.output));metrics=validation["validation"];manifest={"schema_version":"1.0","model_version":args.version,"status":"ready_experimental","dataset":"IDRiD localization training partition","input_size":size,"coordinate_order":validation["coordinate_order"],"confidence_definition":validation["confidence_definition"],"disc_error_scale":metrics["disc_p95_normalized_error"],"fovea_error_scale":metrics["fovea_p95_normalized_error"],"validation":metrics,"official_test_used":False,"checkpoint_sha256":checkpoint_digest,"validation_sha256":sha256(args.validation),"onnx_sha256":sha256(args.output),"checkpoint_epoch":state["epoch"]}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps({"output":str(args.output),"sha256":manifest["onnx_sha256"]}))


if __name__=="__main__":main()
