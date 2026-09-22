"""Optional local ONNX runtimes for learned retinal quality/structure modules."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw


def _load(name:str)->tuple[ort.InferenceSession|None,dict[str,object]]:
    value=os.getenv(name)
    if not value:return None,{}
    path=Path(value);manifest=path.with_suffix(".manifest.json")
    if not path.exists() or not manifest.exists():raise FileNotFoundError(f"{name} requires both ONNX and manifest files")
    return ort.InferenceSession(str(path),providers=["CPUExecutionProvider"]),json.loads(manifest.read_text())


def _data_uri(image:Image.Image)->str:
    image=image.convert("RGB");image.thumbnail((900,900),Image.Resampling.LANCZOS);output=io.BytesIO();image.save(output,format="JPEG",quality=88,optimize=True)
    return "data:image/jpeg;base64,"+base64.b64encode(output.getvalue()).decode("ascii")


def _processed_fundus(image:Image.Image,size:int)->Image.Image:
    rgb=np.asarray(image.convert("RGB"));visible=rgb.max(axis=2)>8;rows=np.flatnonzero(visible.any(axis=1));columns=np.flatnonzero(visible.any(axis=0))
    if rows.size and columns.size:
        padding=round(min(rgb.shape[:2])*0.01);image=Image.fromarray(rgb[max(0,int(rows[0])-padding):min(rgb.shape[0],int(rows[-1])+padding+1),max(0,int(columns[0])-padding):min(rgb.shape[1],int(columns[-1])+padding+1)])
    side=max(image.size);canvas=Image.new("RGB",(side,side));canvas.paste(image,((side-image.width)//2,(side-image.height)//2));return canvas.resize((size,size),Image.Resampling.LANCZOS)


def _segmentation_overlay(image:Image.Image,probabilities:np.ndarray,colors:list[tuple[int,int,int]])->str:
    display=image.convert("RGB");display.thumbnail((900,900),Image.Resampling.LANCZOS);rgb=np.asarray(display,dtype=np.float32);color=np.zeros_like(rgb);alpha=np.zeros(rgb.shape[:2],dtype=np.float32)
    for channel,tint in enumerate(colors):
        mask=Image.fromarray((probabilities[channel]*255).astype(np.uint8)).resize(display.size,Image.Resampling.BILINEAR);values=np.asarray(mask,dtype=np.float32)/255.0;selected=values>=0.5
        for index,value in enumerate(tint):color[...,index]=np.where(selected,np.maximum(color[...,index],value),color[...,index])
        alpha=np.maximum(alpha,np.where(selected,0.55,0.0))
    combined=np.clip(rgb*(1-alpha[...,None])+color*alpha[...,None],0,255).astype(np.uint8);return _data_uri(Image.fromarray(combined))


class AuxiliaryRuntime:
    def __init__(self)->None:
        self.lesion_session,self.lesion_manifest=_load("LESION_MODEL_PATH")
        self.vessel_session,self.vessel_manifest=_load("VESSEL_MODEL_PATH")
        self.localization_session,self.localization_manifest=_load("LOCALIZATION_MODEL_PATH")

    def analyze(self,image:Image.Image,fundus_tensor:Callable[[Image.Image,int,dict[str,object]],np.ndarray])->tuple[dict[str,object],dict[str,object]]:
        structures:dict[str,object]={"status":"not_trained","vessels":{"status":"not_trained"},"optic_disc":{"status":"not_trained"},"fovea":{"status":"not_trained"}}
        lesions:dict[str,object]={"status":"not_trained","experimental":True,"model_version":None,"overlay_image":None,"items":[]}
        ready=False
        if self.lesion_session is not None:
            manifest=self.lesion_manifest;size=int(manifest["input_size"]);settings=dict(manifest.get("preprocessing",{"enhancement":"none","color_normalization":False,"resampling":"bilinear"}));logits=self.lesion_session.run(None,{"image":fundus_tensor(image,size,settings)})[0][0];probabilities=1/(1+np.exp(-np.clip(logits,-60,60)));threshold=float(manifest["threshold"]);items=[]
            for name,probability in zip(manifest["classes"],probabilities):
                selected=probability>=threshold;items.append({"type":name,"status":"experimental","pixel_fraction":round(float(selected.mean()),6),"mean_probability_above_threshold":round(float(probability[selected].mean()),5) if selected.any() else None})
            lesions={"status":"ready","experimental":True,"model_version":manifest["model_version"],"overlay_image":_segmentation_overlay(_processed_fundus(image,size),probabilities,[(255,65,54),(255,149,0),(255,214,10),(175,82,222)]),"items":items};ready=True
        if self.vessel_session is not None:
            manifest=self.vessel_manifest;size=int(manifest["input_size"]);settings=dict(manifest.get("preprocessing",{"enhancement":"none","color_normalization":False,"resampling":"bilinear"}));logits=self.vessel_session.run(None,{"image":fundus_tensor(image,size,settings)})[0][0];probability=1/(1+np.exp(-np.clip(logits,-60,60)));selected=probability[0]>=float(manifest["threshold"])
            structures["vessels"]={"status":"ready","model_version":manifest["model_version"],"pixel_fraction":round(float(selected.mean()),6),"overlay_image":_segmentation_overlay(_processed_fundus(image,size),probability,[(52,199,89)])};ready=True
        if self.localization_session is not None:
            manifest=self.localization_manifest;size=int(manifest["input_size"]);side=max(image.size);canvas=Image.new("RGB",(side,side));canvas.paste(image,((side-image.width)//2,(side-image.height)//2));resized=canvas.resize((size,size),Image.Resampling.LANCZOS);array=np.asarray(resized,dtype=np.float32).transpose(2,0,1)/255.0;mean=np.asarray((0.485,0.456,0.406),dtype=np.float32)[:,None,None];std=np.asarray((0.229,0.224,0.225),dtype=np.float32)[:,None,None]
            coordinates,log_variance=self.localization_session.run(None,{"image":((array-mean)/std)[None]});coordinates=coordinates[0];sigma=np.exp(0.5*log_variance[0]);disc_sigma=float(sigma[:2].mean());fovea_sigma=float(sigma[2:].mean());disc_conf=float(np.exp(-disc_sigma/max(float(manifest["disc_error_scale"]),1e-6)));fovea_conf=float(np.exp(-fovea_sigma/max(float(manifest["fovea_error_scale"]),1e-6)))
            overlay=canvas.copy();draw=ImageDraw.Draw(overlay);radius=max(5,side//90);points=[("optic_disc",coordinates[:2],disc_conf,(255,214,10)),("fovea",coordinates[2:],fovea_conf,(52,199,89))]
            for _,point,_,color in points:
                x=float(point[0])*side;y=float(point[1])*side;draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(2,radius//3))
            structures["optic_disc"]={"status":"ready","model_version":manifest["model_version"],"x":round(float(coordinates[0]),5),"y":round(float(coordinates[1]),5),"confidence":round(disc_conf,5)};structures["fovea"]={"status":"ready","model_version":manifest["model_version"],"x":round(float(coordinates[2]),5),"y":round(float(coordinates[3]),5),"confidence":round(fovea_conf,5)};structures["localization_overlay_image"]=_data_uri(overlay);ready=True
        structures["status"]="ready" if ready else "not_trained"
        return structures,lesions
