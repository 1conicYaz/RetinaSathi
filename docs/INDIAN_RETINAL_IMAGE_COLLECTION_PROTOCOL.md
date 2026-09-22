# RetinaSathi Indian Retinal Image Collection Protocol

## Purpose

This protocol is for a prospective Indian pilot that evaluates RetinaSathi on images from real screening cameras. It is not permission to copy hospital images immediately. The hospital investigator and Institutional Ethics Committee (IEC) must approve the protocol, consent process, storage, access, and intended AI use first.

## First meeting with the doctor or hospital

Ask for agreement on these points before requesting any image:

1. Who is the hospital principal investigator responsible for the data?
2. Is this retrospective research, prospective research, or a quality-improvement activity?
3. Is IEC review or an IEC waiver required? Obtain the written decision.
4. Will patients give specific consent for AI research and future model development?
5. Can de-identified images leave the hospital? If yes, under what data-sharing agreement?
6. Who owns the images and labels, and who may publish results?
7. How long may RetinaSathi retain the data, and how must deletion be documented?
8. Which ophthalmologists will grade the images and adjudicate disagreements?
9. Which camera models, fields of view, and capture settings will be used?
10. What referral definition will the hospital use for referable DR and suspected DME?

Do not collect or copy images until these answers and approvals are documented.

## Recommended pilot size

### Phase A: workflow and image-quality pilot

- 30–50 consenting participants.
- Capture both eyes when clinically appropriate.
- Use this phase to test naming, upload, image quality, grading forms, and removal of identifiers.
- Do not train on Phase A while procedures are still changing.

### Phase B: independent Indian evaluation

- Aim for 300–500 participants and approximately 600–1,000 eye image-sets.
- Include consecutive screening participants so the class distribution reflects real practice.
- Keep this cohort completely outside training and threshold selection.
- Report patient-level confidence intervals and camera-specific results.

### Phase C: Indian model-development cohort

- Build a separate cohort after the evaluation protocol is stable.
- Aim for at least 1,500–3,000 gradable image-sets across multiple sites or cameras.
- Deliberately recruit enough confirmed Grade 1, Grade 3, and Grade 4 cases because these are uncommon and are V3-0 weaknesses.
- Keep patients, not individual images, separated between training, calibration, and evaluation.

These are practical development targets, not a formal sample-size calculation. A statistician should calculate the final sample size from the desired sensitivity confidence interval and expected disease prevalence.

## Image capture

For each eye, prefer two 45-degree colour fundus fields when the camera and clinical workflow support them:

1. Macula-centred image.
2. Optic-disc-centred image.

If only one field is available, record that limitation. Capture personnel should check focus, illumination, field position, visible macula, visible optic disc, and artefacts before the participant leaves. Record every failed or ungradable capture and its reason instead of silently deleting it.

Do not mix OCT images into the colour-fundus classifier directory. Store OCT separately. A colour photograph may suggest DME risk, but clinical DME confirmation may require OCT or examination.

## File naming

Use this pattern:

`RS_<site>_<participant>_<visit>_<eye>_<field>_<sequence>.jpg`

Example:

`RS_H01_P000123_V1_R_MAC_01.jpg`

Allowed eye values: `L`, `R`.

Allowed field values: `MAC`, `DISC`, `OTHER`.

Never put a patient name, phone number, date of birth, address, hospital registration number, Aadhaar number, or diagnosis in a filename.

The hospital should retain the participant-code-to-identity mapping. RetinaSathi should not receive that mapping.

## Minimum metadata

Complete one manifest row for every image:

- Study image ID and participant code.
- Site and visit code.
- Left or right eye.
- Macula-centred, disc-centred, or other field.
- Camera manufacturer, model, camera type, resolution, and field of view.
- Whether dilation was used.
- Image quality and ungradable reason.
- Grader 1 and grader 2 ICDR grades.
- Adjudicated ICDR grade when graders disagree.
- Referable-DR decision.
- Suspected DME referral and whether OCT/examination confirmed it.

Age band, sex, diabetes duration, and HbA1c may help subgroup evaluation, but collect only fields approved by the IEC and needed by the protocol. Do not feed clinical metadata into the model unless that model is separately specified and validated.

## Clinical labels

Use the same five-grade definition across the hospital, dataset, model, interface, and presentation:

- 0: No DR.
- 1: Mild NPDR.
- 2: Moderate NPDR.
- 3: Severe NPDR.
- 4: Proliferative DR.

Two qualified graders should grade independently. If they disagree, a senior ophthalmologist should adjudicate. Keep the original two grades; do not overwrite them with the final grade.

Record image quality separately from disease. An ungradable image is not Grade 0.

Record DME separately from DR grade. Do not infer “no DME” simply because the photograph lacks obvious exudates.

## Safe transfer into RetinaSathi storage

The hospital should export only the approved de-identified package:

1. Retinal image files with coded filenames.
2. Image manifest without direct identifiers.
3. Grading file without direct identifiers.
4. A signed data-transfer note containing image count, checksum, approval reference, transfer date, and sender/receiver.

Before importing:

1. Verify the approval and data-sharing conditions.
2. Run a filename and metadata check for prohibited identifiers.
3. Quarantine files with unexpected formats or embedded identifiers.
4. Compute SHA-256 checksums.
5. Confirm image count against the transfer note.
6. Make a read-only raw copy.
7. Work only from a de-identified derivative copy.

Do not upload clinical images to GitHub, public cloud storage, messaging groups, public InsForge buckets, or Hugging Face datasets.

## Folder structure outside Git

```text
Indian_Clinical_Pilot/
├── 00_approvals_and_data_agreement/
├── 01_raw_restricted_read_only/
├── 02_deidentified_images/
│   ├── H01/
│   └── H02/
├── 03_metadata/
├── 04_grading/
├── 05_quality_control/
├── 06_frozen_evaluation_manifest/
└── 99_transfer_checksums/
```

Access to `01_raw_restricted_read_only` should be limited to named, approved people. The model-training repository should contain manifests and code, not clinical image files.

## Acceptance checklist for each image-set

- Consent/waiver and approved study are documented.
- Filename contains only study codes.
- Participant code is present and consistent.
- Eye and field are known.
- Camera metadata is present.
- Macula and optic disc coverage are recorded.
- Capture quality is graded.
- Two clinical grades or a documented reason for one grader are present.
- Disagreements are adjudicated.
- DR and DME fields are separate.
- No patient identity or facial/anterior photographs are included unless explicitly approved.
- Checksum is recorded.

## What to tell the doctor

“Our current model is good at ranking referable diabetic retinopathy, but it still misses some disease when the camera or dataset changes, and it is weak at separating mild, moderate, and proliferative grades. We need an ethically approved, de-identified Indian validation set that remains outside training. We would like two independent grades, adjudication, image-quality labels, and camera information. The system will remain screening support with mandatory human review.”

## References

- ICMR, *Ethical Guidelines for Application of Artificial Intelligence in Biomedical Research and Healthcare* (2023).
- Ministry of Electronics and Information Technology, *Digital Personal Data Protection Rules, 2025* and enforcement materials.
- NHS Diabetic Eye Screening Programme, *Guidance on fundus image quality and when adequate images cannot be taken*, updated 11 March 2026. This is an operational image-quality reference; local Indian clinical and IEC requirements remain controlling.
