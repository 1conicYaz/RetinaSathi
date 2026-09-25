# RetinaSathi architecture

RetinaSathi combines a React web application, a Python inference service, private InsForge records, a MATLAB image-analysis prototype, and a SimEvents clinic simulation. Read this diagram as the **local V3.4 candidate** path. The public V1 backup and older V2 research modules have separate model contracts.

```mermaid
flowchart TD
    Camera[Fundus photograph] --> Quality{Deterministic quality gate}
    Quality -->|ungradeable| Recapture[Retake; no DR decision]
    Quality -->|gradeable| Prep[Crop, pad, enhance, normalize]
    Prep --> V34[Frozen V3.4 DINOv2 ONNX]
    V34 --> Referral[Calibrated referral score]
    V34 --> Grade[DR grade 0–4 probabilities]
    Referral --> Review[Human review]
    Grade --> Review
    Review --> Report[Screening support report]
    Report --> InsForge[Private owner-scoped records]
    Prep -. optional research view .-> Lesions[V3.2 experimental lesion candidate masks]
    Lesions -. clinician confirmation required .-> Review
```

## Runtime modes

```mermaid
flowchart LR
    Web[React PWA] --> Router{Inference mode}
    Router -->|local| Local[Local FastAPI + selected model]
    Router -->|cloud| Proxy[Authenticated InsForge function]
    Proxy --> Cloud[Protected cloud inference API]
    Router -->|auto| Probe[Try local, then cloud]
    Probe --> Local
    Probe --> Proxy
    Probe -->|neither| Unavailable[No assessment / explicit unavailable state]
    Local -. same frozen ONNX .-> MATLAB[MATLAB prototype]
```

## District workflow

```mermaid
flowchart LR
    Arrival[Patient arrivals] --> Acquire[Camera acquisition queue]
    Acquire --> QGate{Quality gate}
    QGate -->|retake| Acquire
    QGate --> Upload[Network/upload delay]
    Upload --> AI[AI server queue]
    AI --> Priority{Risk/uncertainty priority}
    Priority --> Review[Reviewer queue]
    Review --> Refer[Referral/follow-up]
```

## Version policy

- `V1`: preserved 224×224 MobileNetV3 multi-task baseline and lightweight public backup.
- `V2`: locked EfficientNet-B3 research artifact with an official-test result and a documented Messidor domain-shift failure. Its Grad-CAM, vessel, and localization outputs are version-specific research features.
- `V3.4`: local frozen DINOv2 candidate with a dedicated referral head and grade outputs. DME is **not assessed**. Its unvalidated attention map remains disabled.
- `V3.2 lesion segmentation`: separate optional four-class candidate overlay. It has real pixel-level training and measured internal Dice, but visible false positives prevent a validated lesion claim.
- Model binaries and medical datasets stay outside Git. Small manifests define hashes, preprocessing, calibration, and status; unavailable modules must say so explicitly.
