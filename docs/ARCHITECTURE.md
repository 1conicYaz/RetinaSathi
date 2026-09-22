# RetinaSathi architecture

RetinaSathi separates image inference, product workflow and operational
simulation. This separation keeps model evidence distinct from service-capacity
estimates.

## Retinal inference

```mermaid
flowchart TD
    Image[Fundus photograph] --> Gate{Quality gate}
    Gate -->|Poor| Stop[Stop inference and request recapture]
    Gate -->|Gradeable| Crop[Retina crop and square padding]
    Crop --> Enhance[392 px Ben Graham enhancement]
    Enhance --> Normalize[ImageNet normalization]
    Normalize --> Encoder[Partially adapted DINOv2-S/14]
    Encoder --> Binary[Referable head]
    Encoder --> Ordinal[Ordinal Grade 0–4 head]
    Encoder --> Nominal[Five-class head]
    Binary --> Calibrate[Saved calibration and threshold]
    Ordinal --> Blend[Calibrated grade blend]
    Nominal --> Blend
    Calibrate --> Decision[Referral and uncertainty policy]
    Blend --> Decision
    Decision --> Human[Mandatory human review]
```

V3.4 returns a five-grade probability distribution and a separately calibrated
referable probability. DME is not assessed. Lesion, vessel, optic-disc and
fovea research experiments are not presented as V3.4 clinical outputs.

The selected seed-26038 model is exported to a fixed ONNX graph. MATLAB verifies
the ONNX SHA-256 recorded in the manifest, reproduces the preprocessing and
calibration policy, and matched 25/25 Python reference decisions in the parity
cohort.

## Application and storage

```mermaid
flowchart LR
    Operator[Operator] --> PWA[React PWA]
    PWA --> Router{Inference route}
    Router -->|Local| API[FastAPI V3.4 runtime]
    Router -->|Cloud backup| Cloud[Lightweight cloud runtime]
    API --> Result[Versioned result contract]
    Cloud --> Result
    Result --> DB[Protected screening record]
    Result --> Private[Private image storage]
    DB --> Reviewer[Reviewer view]
    Reviewer --> Review[Append-only review]
    Review --> Report[Printable screening-support report]
```

Authentication, private storage and owner-scoped database policies are handled
through InsForge. The client stores a pseudonymous patient reference rather
than a patient name. An optional seven-day device-local retry queue is available
for temporary connectivity failures and is intended only for secured devices.

## SimEvents operational model

```mermaid
flowchart LR
    Arrival[Patient arrivals] --> CQ[Capture queue]
    CQ --> Camera[Fundus camera server]
    Camera --> Quality{Quality gate}
    Quality -->|Poor| Recapture[Recapture server]
    Recapture --> CQ
    Quality -->|Accepted| Network[Network transfer]
    Network --> AQ[AI queue]
    AQ --> AI[AI service-time server]
    AI --> Route{Decision route}
    Route -->|Routine| Routine[Routine outcome]
    Route -->|Referable or uncertain| RQ[Clinical review queue]
    RQ --> Reviewer[Human reviewer]
    Reviewer --> Complete[Reviewed outcome]
```

One entity represents one screening visit. The workflow uses configured arrival
rates, service times, capacities, image size, bandwidth and routing rates. The
AI server represents measured/configured station time; it does not execute ONNX
for each synthetic visit.

The seven scenarios test baseline operation, constrained bandwidth, increased
load, an additional camera, an additional reviewer, priority review and a
district configuration. The automatic verifier checks accounting, utilization,
network behaviour, resource sensitivity and annual-equivalent capacity.

## Safety boundaries

- Poor-quality images do not receive a DR prediction.
- Every output requires human review.
- Attention is model influence, not a lesion map.
- Missing modules are reported as unavailable or not assessed.
- Simulation figures are planning estimates, not hospital observations.
- Source validation and software parity do not establish clinical readiness.
