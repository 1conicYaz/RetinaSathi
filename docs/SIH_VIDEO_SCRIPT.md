# RetinaSathi SIH video script

Target length: 5–7 minutes. Record the real prototype and keep credentials, keys and private images off screen.

1. **Problem, 20 seconds.** Explain that DR screening needs accessible image quality checks, referral support and specialist review. Call RetinaSathi a research prototype.
2. **Architecture, 30 seconds.** Show `fundus image → quality → V3.4 → referral/grade → human review → history` and the authenticated InsForge-to-Azure path.
3. **Sign in, 15 seconds.** Open <https://69exmaqk.insforge.site/> and sign in with a demo account.
4. **Valid image, 60 seconds.** Upload an approved public retinal fixture. Show model identity, assessment state, referral score, threshold, grade probabilities, quality and required human review.
5. **Safety case, 35 seconds.** Upload the poor-quality fixture. Show that prediction stops with `retake_required`; no grade, negative referral or Routine action appears.
6. **History, 25 seconds.** Open History, reload the page and show that the assessment state and provenance persist.
7. **XAI honesty, 15 seconds.** Say: “V3.4 explanation is disabled because the previous map did not pass technical validation. We prefer no heatmap to misleading evidence.”
8. **Azure security, 25 seconds.** Show the architecture diagram, never secrets. Explain authentication, server-side inference key, private storage and scale-to-zero cold-start tradeoff.
9. **MATLAB, 30 seconds.** Show the parity command and passing result. Say it proves implementation consistency, not clinical accuracy.
10. **SimEvents, 35 seconds.** Show queues and scenario dashboard. Explain camera, network and reviewer bottlenecks. Label every throughput number as simulation.
11. **Evidence, 30 seconds.** State retrospective selected-seed sensitivity 92.22% and specificity 91.36% on the reused patient-separated DeepDRiD source holdout; state that exact grading and independent validation remain weaker/missing.
12. **Roadmap, 30 seconds.** Finish with independent intended-camera evaluation, stronger quality/OOD and governed review, then validated explanations/pathology and edge optimization.

Closing line:

> RetinaSathi demonstrates a reproducible, human-in-the-loop screening-support pathway today, while keeping clinical validation and patient safety as explicit next steps.
