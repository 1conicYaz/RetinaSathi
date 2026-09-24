# RetinaSathi SIH demo runbook

## Before the presentation

```bash
git status --short
npm ci
npm test
npm run lint
npm run typecheck
npm run build
curl -fsS https://retinasathi-v34.proudcoast-d5c4449c.uaenorth.azurecontainerapps.io/health
curl -fsS https://retinasathi-v34.proudcoast-d5c4449c.uaenorth.azurecontainerapps.io/model-card
```

Confirm model `classifier-v3.4-seed26038`, architecture `dinov2_vits14`, input 392 and threshold `0.20732617378234863`. Never show keys or `.env` files.

## Demonstration order

Present MATLAB and SimEvents before the supporting website. If time is limited,
do not cut the MathWorks components before the web demonstration.

## MATLAB demonstration

```bash
./scripts/verify_v3_4_matlab.sh
```

Expected verified summary: `PASS: 8 MATLAB tests, 25 parity cases, grade 1.000, referral 1.000`.

Say: “This proves implementation parity with the frozen artifact, not clinical validation.”

## SimEvents demonstration

In MATLAB:

```matlab
cd('simulink')
report = verify_simevents_workflow(true, "results/simevents");
launch_simevents_demo
```

Show queues and the scenario dashboard before the website. State that annual
equivalents are simulation estimates under declared assumptions.

## Supporting web/cloud demonstration

1. Open <https://69exmaqk.insforge.site/>.
2. Sign in with the demo account without exposing credentials.
3. Upload the approved public fixture `runs/demo_cases/referable.jpg` using patient reference `SIH-DEMO-REF`.
4. Show quality, explicit assessment state, dedicated referral score, threshold, grade probabilities and V3.4 identity.
5. Explain that every result requires human review.
6. Open History, confirm the result exists, reload and confirm the same state remains.
7. Upload `runs/demo_cases/poor_quality.jpg` as `SIH-DEMO-RETAKE`.
8. Show `Retake required / not assessed`, with null grade/referral decision and no Routine label.
9. State that V3.4 XAI, DME and lesion modules are unavailable.

## Local V3.4 fallback

The ONNX file must match the manifest hash.

```bash
./scripts/run_v3_4_local.sh
```

In another terminal:

```bash
cp .env.example .env.local
# Set VITE_INFERENCE_MODE=local in .env.local
npm run dev
```

Check:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/model-card
```

## Recovery rules

- Azure cold start: wait once and retry; explain scale-to-zero.
- Cloud unavailable: switch to the verified local V3.4 path.
- Never substitute a different model without showing its identity.
- Never use private patient images in the demo.
- If a step fails, state it plainly and continue with stored evidence.
