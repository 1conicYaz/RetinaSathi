# Dataset strategy and EyePACS admission decision

RetinaSathi uses a problem-driven dataset policy. Dataset size alone is not an
admission criterion. Every source must pass provenance, usage-rights, label,
patient-grouping, duplicate, image-quality and domain-relevance checks.

## Verified V3.4 roles

| Dataset | Verified role | Used in V3.4? | Important limitation |
|---|---|---:|---|
| APTOS 2019 | Indian development pool | Yes | Patient identifiers are unavailable in the verified labels |
| IDRiD grading training partition | Indian development pool | Yes | Small and imbalanced; grading CSV lacks patient IDs |
| DeepDRiD training partition | Grouped development pool | Yes | Chinese acquisition domain |
| DeepDRiD validation partition | Patient-separated source validation | Evaluation only | Reused during model development; not final independent clinical evidence |
| IDRiD lesion/anatomy tasks | Separate lesion/localization research | No classification role | Same images can occur across task folders |
| DRIVE | Vessel research | No | Not a DR grading dataset |
| Messidor | Historical V2 evaluation | No | Grade 0–3 scale is not interchangeable with ICDR 0–4 |
| EyePACS/Kaggle 2015 | Candidate controlled experiment | No | Competition terms, label noise/quality and overlap require audit |

## Why V3.4 did not use EyePACS

### Verified reasons

- The audited V3 manifest contains APTOS, IDRiD and DeepDRiD; no EyePACS row is
  present in V3.4 training, calibration or source validation.
- The local inventory recorded only an approximately 82 GiB compressed archive;
  no canonical extracted image/label tree was admitted to the pipeline.
- Therefore no repository evidence establishes archive integrity, image-label
  pairing, class counts, preprocessing success, duplicate status or overlap
  against the current sources.
- EyePACS is governed by Kaggle competition rules and cannot be redistributed.
  The official data page explicitly says that images use subject IDs with
  left/right eye suffixes and that data sharing outside Kaggle is prohibited.

### Corrected facts

- Patient grouping is not intrinsically unavailable. The official naming scheme
  is `[subject_id]_[left/right]`, so both eyes can be grouped by subject after
  extraction and label verification.
- The public labelled training set contains 35,126 images with Grade 0–4 labels.
  Published counts are 25,810 / 2,443 / 5,292 / 873 / 708, showing severe class
  imbalance. EyePACS adds many Grade-2 examples, but relatively few Grades 3–4.
- Images were captured under varied conditions. This diversity may help camera
  robustness, but it also introduces quality variation and possible domain
  shift. More data does not automatically improve Indian intended-camera use.

Official source: <https://www.kaggle.com/c/diabetic-retinopathy-detection/data>.
The published class counts are reproduced in a peer-reviewed multi-dataset
study: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7579670/>. A separate clinical
study relabelled an EyePACS validation subset with three retina specialists,
which supports treating original labels as development labels rather than a
final adjudicated clinical reference:
<https://jamanetwork.com/journals/jamaophthalmology/fullarticle/2810431>.

## Decision: EXPERIMENT FIRST

EyePACS is a plausible future development source, especially for testing whether
additional Grade-2 and acquisition diversity improve five-grade performance.
It should not be merged directly into V3.4 or used as a new test set.

### Research question

Can carefully curated EyePACS development data improve Grade-2 recall and
five-grade macro-F1 without reducing Indian-source referral sensitivity,
specificity or calibration?

### Admission gate

1. Mount the external SSD and hash every archive part and label file.
2. Confirm the exact Kaggle source and accepted rules for this research use.
3. Extract only to controlled external storage; never copy images into Git.
4. Validate every labelled filename, JPEG header and subject/eye pair.
5. Group both eyes by subject before any split.
6. Audit exact hashes and perceptual near-duplicates within EyePACS and against
   APTOS, IDRiD and DeepDRiD before assigning splits.
7. Produce quality, dimension and Grade 0–4 distributions; quarantine corrupt,
   ungradable and unresolved-label cases without deleting raw evidence.
8. Freeze the current V3.4 evaluation protocol and threshold.
9. Train two otherwise identical challengers: baseline without EyePACS and
   experiment with admitted EyePACS development data.
10. Compare referral sensitivity/specificity, AUROC/AUPRC, QWK, macro-F1,
    per-grade recall, calibration and performance by source/camera quality.
11. Adopt only if gains repeat across seeds and do not weaken the safety gate.

The external SSD was not mounted during the 24 September 2026 repository review,
so the physical archive could not be re-audited in that review. This document
does not claim the archive is currently complete or usable.

## Future independent evidence

The preferred final evidence is a prospectively governed Indian intended-camera
cohort with patient-level separation, multiple sites, independent qualified
graders, adjudication, representative poor-quality cases and a model frozen
before evaluation. EyePACS cannot substitute for that evidence.
