# RetinaSathi SIH video script

Target length: 5–7 minutes. Record the real prototype and keep credentials, keys and private images off screen.

1. **Problem, 20 seconds.** Explain that DR screening needs accessible image quality checks, referral support and specialist review. Call RetinaSathi a research prototype.
2. **MathWorks architecture, 30 seconds.** Show `fundus image → quality → V3.4 ONNX → MATLAB calibration/referral + grade → human review`.
3. **MATLAB, 45 seconds.** Run one image, then show the 8-test and 25/25 parity summary. Say it proves implementation consistency, not clinical accuracy.
4. **SimEvents, 50 seconds.** Show capture, recapture, network, AI and reviewer queues. Compare high load with an added camera and label every throughput number as simulation.
5. **Web interface, 15 seconds.** Explain that the web application is the accessible deployment interface, then sign in with a demo account.
6. **Valid image, 60 seconds.** Upload an approved public retinal fixture. Show model identity, assessment state, referral score, threshold, grade probabilities, quality and required human review.
7. **Safety case, 35 seconds.** Upload the poor-quality fixture. Show that prediction stops with `retake_required`; no grade, negative referral or Routine action appears.
8. **History, 25 seconds.** Reload History and show that the assessment state and provenance persist.
9. **XAI honesty, 15 seconds.** Say: “V3.4 explanation is disabled because the previous map did not pass technical validation. We prefer no heatmap to misleading evidence.”
10. **Azure security, 25 seconds.** Show the architecture diagram, never secrets. Explain authentication, server-side inference key, private storage and scale-to-zero cold-start tradeoff.
11. **Evidence, 30 seconds.** State retrospective selected-seed sensitivity 92.22% and specificity 91.36% on the reused patient-separated DeepDRiD source holdout; state that exact grading and independent validation remain weaker/missing.
12. **Roadmap, 30 seconds.** Finish with ophthalmologist workflow research, independent intended-camera evaluation, quality/OOD, validated XAI, controlled EyePACS research, distillation and secure edge/offline evaluation.

Closing line:

> RetinaSathi demonstrates a reproducible, human-in-the-loop screening-support pathway today, while keeping clinical validation and patient safety as explicit next steps.
