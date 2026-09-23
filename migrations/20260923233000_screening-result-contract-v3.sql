-- Preserve safety-critical inference semantics and immutable runtime provenance.
ALTER TABLE public.screenings
  ADD COLUMN assessment_state TEXT,
  ADD COLUMN referral_score REAL CHECK (referral_score BETWEEN 0 AND 1),
  ADD COLUMN referral_threshold REAL CHECK (referral_threshold BETWEEN 0 AND 1),
  ADD COLUMN referral_decision BOOLEAN,
  ADD COLUMN model_sha256 TEXT CHECK (model_sha256 IS NULL OR model_sha256 ~ '^[a-f0-9]{64}$'),
  ADD COLUMN config_sha256 TEXT CHECK (config_sha256 IS NULL OR config_sha256 ~ '^[a-f0-9]{64}$'),
  ADD COLUMN calibration_version TEXT,
  ADD COLUMN explanation_status TEXT,
  ADD COLUMN operation_id UUID,
  ADD COLUMN result_snapshot JSONB;

UPDATE public.screenings
SET assessment_state = CASE
      WHEN status = 'needs_retake' OR quality_label = 'poor' THEN 'retake_required'
      WHEN status = 'failed' OR dr_grade IS NULL THEN 'not_assessed'
      WHEN referable_dr IS TRUE THEN 'assessed_referable'
      WHEN uncertainty_label = 'high' THEN 'uncertain'
      ELSE 'assessed_non_referable'
    END,
    referral_decision = CASE
      WHEN status IN ('needs_retake', 'failed') OR quality_label = 'poor' OR dr_grade IS NULL THEN NULL
      ELSE referable_dr
    END,
    referable_dr = CASE
      WHEN status IN ('needs_retake', 'failed') OR quality_label = 'poor' OR dr_grade IS NULL THEN NULL
      ELSE referable_dr
    END,
    operation_id = gen_random_uuid();

ALTER TABLE public.screenings
  ALTER COLUMN assessment_state SET NOT NULL,
  ALTER COLUMN operation_id SET NOT NULL,
  ADD CONSTRAINT screenings_assessment_state_check CHECK (
    assessment_state IN ('assessed_referable', 'assessed_non_referable', 'not_assessed', 'retake_required', 'uncertain')
  ),
  ADD CONSTRAINT screenings_not_assessed_has_no_decision CHECK (
    assessment_state NOT IN ('not_assessed', 'retake_required')
    OR (dr_grade IS NULL AND referral_decision IS NULL AND referable_dr IS NULL)
  ),
  ADD CONSTRAINT screenings_referral_fields_agree CHECK (
    referral_decision IS NULL OR referable_dr = referral_decision
  );

CREATE UNIQUE INDEX screenings_user_operation_unique
  ON public.screenings (user_id, operation_id);

COMMENT ON COLUMN public.screenings.assessment_state IS
  'Safety state that distinguishes a valid negative result from not-assessed, retake, and uncertain cases.';
COMMENT ON COLUMN public.screenings.result_snapshot IS
  'Versioned text/provenance snapshot; image and explanation data URIs are intentionally excluded.';
