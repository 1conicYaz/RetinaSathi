# Documentation guide

The repository separates measured evidence from plans so readers can see what
works today and what still requires validation.

## Start here

1. [Complete project explanation](COMPLETE_PROJECT_EXPLANATION.md) — plain-language
   walkthrough from the clinical problem to deployment and future work.
2. [Architecture](ARCHITECTURE.md) — implemented system boundaries and data flow.
3. [Model selection](MODEL_SELECTION_AND_APPROACH_REPORT.md) — why EfficientNet,
   DINOv2 and RETFound were compared.
4. [V3.4 results](V3_4_RESULTS.md) and [three-seed stability](V3_4_THREE_SEED_STABILITY.md)
   — retrospective source-validation evidence.
5. [V3 data card](V3_DATA_CARD.md) — sources, splits, duplicates and limitations.

## Engineering verification

- [MATLAB parity](V3_4_MATLAB_PARITY_REPORT.md) — Python, ONNX and MATLAB
  implementation consistency.
- [SimEvents results](SIMULATION_RESULTS.md) — operational scenarios under
  declared assumptions.
- [Azure deployment verification](AZURE_DEPLOYMENT_VERIFICATION.md) — deployed
  model identity, access path and known gaps.
- [SIH, MATLAB and Simulink audit](SIH_MATLAB_SIMULINK_FULL_AUDIT.md) — testing
  instructions and requirement status.

## Presentation and next steps

- [Judge questions and answers](SIH_JUDGE_QA.md)
- [Suggested six-slide deck](SIH_SIX_SLIDE_CONTENT.md)
- [Demo video script](SIH_VIDEO_SCRIPT.md)
- [Clinical pilot plan](V3_4_CLINICAL_PILOT_AND_APP_INTEGRATION.md)
- [Future roadmap](FUTURE_ROADMAP.md)

Source validation, software verification, simulation estimates and prospective
clinical evidence are different forms of evidence. The documents do not treat
one as proof of another.
