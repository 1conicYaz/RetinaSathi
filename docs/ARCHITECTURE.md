# RetinaSathi architecture

RetinaSathi separates MATLAB-compatible image inference, SimEvents operational
simulation and the supporting web deployment. This keeps model evidence,
capacity estimates and product behavior distinct.

## System view

```mermaid
flowchart TD
    Camera[Fundus camera] --> Quality{Quality and recapture}
    Quality -->|Gradeable| V34[RetinaSathi V3.4 ONNX]
    V34 --> MATLAB[MATLAB preprocessing, calibration and parity]
    MATLAB --> Decision[Referral score and Grade 0–4]
    Decision --> Human[Mandatory human review]
    Human --> Report[Screening-support report]
    Workflow[Simulink / SimEvents operational model] -. models queues and resources .-> Camera
    Workflow -. capacity planning .-> Human
    Web[React + InsForge + Azure interface] -. supporting deployment path .-> V34
    Web -. protected workflow .-> Report
```

The diagram contains two related paths. Solid arrows show one-image screening.
Dotted arrows show simulation or deployment support; SimEvents does not execute
the image model for each synthetic visit.

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

## Supporting application and storage

```mermaid
flowchart LR
    Operator[Operator] --> PWA[React PWA]
    PWA --> Auth[InsForge authentication]
    Auth --> Fn[Authenticated server function]
    Fn --> Azure[Protected Azure V3.4 ONNX service]
    Azure --> Result[Versioned result contract]
    Result --> DB[Protected screening record]
    Result --> Private[Private image storage]
    DB --> Reviewer[Reviewer view]
    Reviewer --> Review[Append-only review]
    Review --> Report[Printable screening-support report]
```

Authentication, private storage and owner-scoped database policies are handled
through InsForge. The client stores a pseudonymous patient reference rather
than a patient name. A local FastAPI runtime is available for development and
demo fallback. The optional seven-day browser retry queue requires stronger
device security and recovery testing before any pilot.

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
