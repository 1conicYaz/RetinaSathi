# RetinaSathi model-selection and machine-learning approach report

**Project:** RetinaSathi — Explainable diabetic-retinopathy screening support for rural India
**Current selected model:** V3.4, partially adapted DINOv2-S/14
**Purpose:** Explain why each model was tested, what was learned, and why the current approach was selected

## 1. Simple summary

We did not try several famous models randomly. Each model answered a different engineering question:

- **MobileNetV3** asked whether a very small model could prove the complete application workflow.
- **EfficientNet-B3** asked how strong a conventional, efficient retinal classifier could become with ordinal learning and better data handling.
- **DINOv2-S, frozen** asked whether a compact general-purpose foundation model already had useful visual features without expensive adaptation.
- **RETFound, frozen** asked whether a large model pretrained specifically on retinal images transferred better.
- **DINOv2-S, partially fine-tuned** asked whether adapting only the final transformer blocks could provide the best balance of sensitivity, specificity, grading, size, and runtime.

The experiments showed that model reputation alone is not enough. RETFound is retinal-specific, but our frozen RETFound adaptation failed badly. EfficientNet detected most referable cases but produced too many false referrals. Frozen DINOv2 was balanced but missed the sensitivity target. Partially adapting DINOv2 produced the strongest overall referral result and passed the declared gate in three separate training runs.

V3.4 should therefore be presented as a **research screening-support candidate for referable DR**, with mandatory human review. It should not be presented as a clinically validated autonomous diagnostic system or as a solved five-grade classifier.

## 2. The problem the models must solve

RetinaSathi has two related outputs:

1. **Screening decision:** Is the image likely to contain referable diabetic retinopathy, meaning Grade 2 or above?
2. **Severity estimate:** Which grade from 0 to 4 best describes the image?

These outputs have different risks. Missing a referable case can delay treatment. Referring too many healthy or mild cases can overload ophthalmologists. Exact five-grade classification is harder because neighbouring grades can look similar and some grades have much less training data.

The model must also handle different cameras, image colours, resolutions, compression, illumination, disease prevalence, and dataset labelling practices. A model that performs well on one dataset can fail on another source.

## 3. Why we used a controlled comparison

Comparing unrelated online accuracy numbers would be misleading because papers often use different datasets, splits, preprocessing, labels, and evaluation rules. We instead kept the RetinaSathi data protocol and prediction heads as consistent as possible and changed one major idea at a time.

This approach gives every experiment a clear question:

| Experiment | Main change | Question |
|---|---|---|
| V3.0 | Multi-source EfficientNet-B3 | Does better, source-aware data alone improve the existing model family? |
| V3.1 | Three heads and joint source/grade sampling | Can direct grade supervision improve rare grades and referral sensitivity? |
| V3.2 | Frozen DINOv2-S backbone | Are compact foundation-model features stronger without backbone training? |
| V3.3 | Frozen RETFound backbone | Does retinal-specific pretraining transfer better as a frozen feature extractor? |
| V3.4 | Final two DINOv2 blocks adapted | Does limited domain adaptation fix the frozen model's sensitivity while preserving specificity? |

Failed experiments were retained because they show which assumptions were wrong and prevent the team from repeating the same mistake.

## 4. MobileNetV3 — lightweight application baseline

### Why it was used

MobileNetV3 is designed for efficient inference on limited hardware. It was useful for proving the complete engineering workflow:

- image upload;
- inference API;
- result storage;
- web interface;
- model versioning;
- local and cloud operation;
- early MATLAB ONNX import.

### What it taught us

The model was small enough for a lightweight cloud deployment, but its five-grade performance was limited. The V1 record reports QWK 0.494 and macro-F1 0.361. Its earlier model-selection history also contained bias and data-leakage limitations, so it is retained as an engineering baseline rather than evidence for clinical use.

### Present role

MobileNetV3 remains useful as a lightweight fallback and interoperability example. It should not replace V3.4 on the basis of speed alone.

## 5. EfficientNet-B3 — strong convolutional control

### Why it was used

EfficientNet scales network depth, width, and image resolution in a controlled way. Convolutional networks are good at local textures and shapes, which matter for retinal abnormalities. EfficientNet-B3 also offers a practical balance between accuracy, memory, and deployment cost.

It served as the **control model**: before claiming that a large foundation model was necessary, we needed to know how far a well-trained conventional model could go.

### V2 role

V2 combined EfficientNet-B3 with ordinal learning. Ordinal learning treats DR grades as an ordered sequence rather than five unrelated names. The historical locked IDRiD result was 82.8% sensitivity and 92.3% specificity. Specificity was useful, but sensitivity did not reach the screening target.

### V3.0 role

V3.0 retrained EfficientNet-B3 under the new multi-source protocol. It produced 87.22% sensitivity and 91.82% specificity. This showed that the data protocol gave a reasonable balance, but sensitivity still failed the source gate.

### V3.1 role

V3.1 added three complementary outputs and joint source/grade-balanced sampling:

- a binary referable-DR head;
- an ordinal Grade 0–4 head;
- a direct five-class head.

It achieved 97.22% sensitivity and QWK 0.834, proving that EfficientNet could detect most referable cases and grade reasonably well. However, specificity fell to 77.73%. In simple terms, it caught many diseased cases but also referred too many non-referable cases.

### What EfficientNet taught us

EfficientNet-B3 was a strong baseline and remains valuable as a future compact student model. Its weakness in V3.1 was the safety/efficiency balance: very high sensitivity came with excessive false referrals.

## 6. DINOv2-S — compact general visual foundation model

### What DINOv2 is

DINOv2 is a vision-transformer foundation model pretrained through self-supervised learning on a large and varied image collection. It was not originally trained to diagnose diabetic retinopathy. Its value is that it learns general representations of shape, texture, structure, and context before seeing RetinaSathi labels.

We selected the small ViT-S/14 variant because it is much more practical than very large foundation models. It uses image patches and transformer attention, giving it a different way of representing the retina from a convolutional EfficientNet.

### Why V3.2 froze the backbone

The first foundation-model test intentionally froze DINOv2 and trained only the RetinaSathi prediction heads. This answered a low-cost question: are its existing features already useful enough?

V3.2 achieved:

- sensitivity: 87.22%;
- specificity: 87.27%;
- AUROC: 0.9412;
- QWK: 0.7440;
- macro-F1: 0.4220.

It was more balanced than V3.1 but missed the required sensitivity. The conclusion was not that DINOv2 was unsuitable. The conclusion was that a completely frozen general-purpose encoder could not adapt enough to retinal grading and source differences.

## 7. RETFound — retinal foundation-model experiment

### What RETFound is

RETFound is a large foundation model pretrained on retinal images using masked-image modelling. It was important to test because retinal pretraining could, in theory, provide disease-relevant features that a general foundation model lacks.

### Why we tested it frozen first

RETFound has roughly 303 million encoder parameters. Fully fine-tuning a model of this size costs much more memory and training time. A frozen feature probe was used first to determine whether its pretrained representation transferred cleanly before paying for a larger campaign.

### What happened in V3.3

The encoder remained frozen and only about 10,000 head parameters were trained. The result was:

- sensitivity: 58.89%;
- specificity: 88.64%;
- AUROC: 0.8584;
- QWK: 0.1412;
- macro-F1: 0.2010.

On the DeepDRiD source, recall was 0% for Grades 1, 2, and 3. The model mainly separated healthy-looking images and some severe cases.

### Why it failed

The experiment identified several likely causes:

1. The huge encoder was completely frozen, so it could not adapt to RetinaSathi's cameras, labels, preprocessing, and grade definitions.
2. The experiment used the CLS token, while the official downstream RETFound recipe uses trainable downstream normalisation and global pooling.
3. Ben Graham enhancement may have created a mismatch with the images used during RETFound pretraining.
4. The small heads could not create useful middle-grade boundaries from fixed features.
5. Class imbalance and cross-dataset camera differences remained difficult.

This result does not prove that RETFound is a poor model. It proves that **our frozen RETFound adaptation was unsuitable**. We rejected it rather than hiding the failure or assuming that a retinal model must automatically win.

### Present role

RETFound is not the selected teacher. A future research branch could test proper partial fine-tuning, official-style pooling, and ordinary RGB preprocessing, but only if compute and new evaluation data justify the work.

## 8. V3.4 — partially adapted DINOv2-S

### Why this was the logical next step

V3.2 was the most balanced compact foundation candidate, but its frozen backbone missed sensitivity. V3.4 therefore changed one major factor: it adapted the final two DINOv2 transformer blocks and final normalisation while keeping the earlier layers frozen.

This provides a compromise:

- early general visual knowledge is preserved;
- final features can adapt to retinal photographs and DR labels;
- training is cheaper and more stable than fully fine-tuning the whole model;
- the model remains much smaller than RETFound;
- the same three prediction heads and source-aware sampling are retained.

### V3.4 architecture

The V3.4 input is a 392 × 392 retinal image prepared using retinal cropping, square padding, Ben Graham enhancement, and ImageNet channel normalisation.

It returns three outputs:

1. **Binary head:** estimates referable versus non-referable DR.
2. **Ordinal head:** learns four ordered boundaries that construct Grade 0–4 probabilities.
3. **Nominal head:** directly learns five grade classes, helping rare grades receive an explicit signal.

The final grade distribution is an equal blend of the calibrated ordinal and nominal probabilities. The referral decision comes from the separately calibrated binary output and its saved threshold.

### Why three heads are useful

The binary head focuses on the most important screening action: whether referral may be needed. The ordinal head understands that confusing Grade 2 with Grade 3 is less severe than confusing Grade 0 with Grade 4. The nominal head still gives each grade its own direct learning target.

This structure avoids forcing one output to solve every problem.

### Repeated results

| Seed | Sensitivity | Specificity | AUROC | AUPRC | QWK | Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 26038 | 92.22% | 91.36% | 0.9756 | 0.9745 | 0.7910 | 0.5339 |
| 26039 | 95.00% | 89.55% | 0.9779 | 0.9758 | 0.8635 | 0.6017 |
| 26040 | 94.44% | 90.91% | 0.9807 | 0.9787 | 0.8195 | 0.5641 |
| **Mean** | **93.89%** | **90.61%** | **0.9781** | **0.9763** | **0.8247** | **0.5666** |

All three runs exceeded the declared DeepDRiD source gate of 90% sensitivity and 85% specificity. The narrow AUROC and AUPRC spread indicates stable referral discrimination across random initialisations.

### Why V3.4 was selected

V3.4 provided the best measured balance:

- V3.1 had higher sensitivity but inadequate specificity.
- V3.2 was balanced but had inadequate sensitivity.
- V3.3 failed to learn the middle grades.
- V3.4 kept both sensitivity and specificity above the source gate in three runs.

Seed 26038 is used for the current application because it was the first predeclared V3.4 run. Choosing seed 26039 only because it later produced the highest observed score would allow the diagnostic source to influence model selection.

## 9. Full comparison

| Version | Model idea | Sensitivity | Specificity | QWK | Macro-F1 | Main conclusion |
|---|---|---:|---:|---:|---:|---|
| V1 | MobileNetV3 lightweight baseline | Historical baseline | Historical baseline | 0.494 | 0.361 | Useful engineering baseline; limited grading evidence |
| V2 | EfficientNet-B3 ordinal classifier | 82.8% | 92.3% | 0.694 on locked IDRiD | Not the same evaluation table | Good specificity; missed sensitivity target |
| V3.0 | Multi-source EfficientNet-B3 | 87.22% | 91.82% | See run record | See run record | Better data protocol alone was insufficient |
| V3.1 | EfficientNet-B3 with three heads | 97.22% | 77.73% | 0.8340 | 0.5341 | Strong sensitivity and grading; too many false referrals |
| V3.2 | Frozen DINOv2-S | 87.22% | 87.27% | 0.7440 | 0.4220 | Balanced foundation candidate; insufficient adaptation |
| V3.3 | Frozen RETFound | 58.89% | 88.64% | 0.1412 | 0.2010 | Frozen adaptation failed badly |
| V3.4 | Partial DINOv2-S, three-seed mean | 93.89% | 90.61% | 0.8247 | 0.5666 | Best balance; selected research candidate |

Metrics from different evaluation stages should not be treated as one leaderboard unless the endpoint and cohort are identical. The V3.1–V3.4 rows are the strongest direct source-validation comparison.

## 10. What remains weak

V3.4 has not solved exact five-grade classification. Three-seed mean recalls were:

| Grade | Meaning | Mean recall |
|---:|---|---:|
| 0 | No DR | 84.67% |
| 1 | Mild NPDR | 53.62% |
| 2 | Moderate NPDR | 51.45% |
| 3 | Severe NPDR | 57.35% |
| 4 | Proliferative DR | 43.33% |

Grade 2 varied considerably across seeds, and Grade 4 had only 20 images in the source-validation set. The supported use is therefore referable-DR screening support followed by human review. The grade should be shown as research information, not an autonomous diagnosis.

The model also does not contain validated heads for DME, lesion segmentation, vessels, optic disc, fovea, or neovascularisation. Model-attention maps show influence, not confirmed lesions.

## 11. Why the overall approach is strong for SIH

The strength is not simply that V3.4 has the highest numbers. The process demonstrates:

- a lightweight working baseline;
- leakage-aware multi-source development;
- a conventional convolutional control;
- general and retinal foundation-model comparisons;
- honest rejection of failed experiments;
- separate referral and severity objectives;
- calibration and a frozen referral threshold;
- repeated-seed stability instead of one lucky run;
- human review and uncertainty routing;
- MATLAB interoperability and Simulink workflow planning;
- explicit limitations and a path to independent Indian validation.

This is more defensible than selecting a famous model, reporting only accuracy, or hiding the classes where the system performs poorly.

## 12. Next model-development steps

1. Freeze V3.4 seed 26038, its preprocessing, calibration, threshold, manifests, and hashes.
2. Complete Python-to-ONNX-to-MATLAB parity before presenting MATLAB V3.4 inference as complete.
3. Run the untouched locked endpoint once after the candidate package is formally frozen.
4. Acquire a new de-identified Indian-camera cohort with patient-level separation and ophthalmologist-reviewed labels.
5. Report performance by camera/source, image quality, and each grade.
6. Improve Grades 2–4 in a new research branch without modifying the frozen V3.4 evidence.
7. Consider a properly adapted RETFound or another challenger only under the same protocol and with sufficient compute.
8. Distil V3.4 into EfficientNet-B3, EfficientNetV2-S, or another compact student only if latency or memory measurements require it.
9. Recalibrate and validate any student independently; it cannot inherit the teacher's evidence automatically.

## 13. Explanation for teammates and judges

Use this short explanation:

> We used several models because each one tested a different idea. EfficientNet-B3 gave us a strong and efficient convolutional baseline. Frozen DINOv2 tested whether general foundation features were already useful. Frozen RETFound tested whether retinal-specific pretraining automatically transferred, and it failed on the middle grades, so we rejected that adaptation. We then partially adapted the final DINOv2 blocks. This kept its general visual knowledge while allowing it to learn our retinal domain. V3.4 was selected because all three repeated runs exceeded our referable-DR sensitivity and specificity gates. Exact five-grade performance remains weaker, so RetinaSathi is screening support with mandatory human review.

If asked why RETFound was not selected:

> RETFound is a strong retinal foundation model, but model quality depends on adaptation, preprocessing, labels, and evaluation data. Our frozen RETFound probe achieved zero recall for Grades 1–3 on the diagnostic source. We recorded the failure instead of selecting it because of its name.

If asked why EfficientNet was not selected:

> EfficientNet-B3 detected most referable cases, but its V3.1 specificity was 77.73%, which would produce too many unnecessary referrals. Partially adapted DINOv2 provided a better sensitivity-specificity balance.

If asked whether V3.4 is clinically ready:

> No. It is a frozen research screening candidate with retrospective source validation. Independent Indian-camera testing and prospective clinician-supervised validation are still required.

## 14. Primary model references

- EfficientNet paper: <https://proceedings.mlr.press/v97/tan19a.html>
- DINOv2 official repository: <https://github.com/facebookresearch/dinov2>
- DINOv2 paper: <https://arxiv.org/abs/2304.07193>
- RETFound official repository: <https://github.com/rmaphoh/RETFound>
- RETFound paper: <https://www.nature.com/articles/s41586-023-06555-x>
