import { z } from 'zod';

export const assessmentStates = [
  'assessed_referable',
  'assessed_non_referable',
  'not_assessed',
  'retake_required',
  'uncertain',
] as const;

export type AssessmentState = typeof assessmentStates[number];

const moduleStatus = z.enum([
  'ready', 'candidate', 'baseline_v1', 'heuristic', 'not_trained', 'data_limited', 'unavailable',
]);

const finiteProbability = z.number().finite().min(0).max(1);

export const screeningResultSchema = z.object({
  api_version: z.literal('2.0'),
  assessment: z.object({ state: z.enum(assessmentStates), reason: z.string().min(1) }),
  model_version: z.string().min(1),
  model_identity: z.object({
    model_name: z.string().min(1),
    model_version: z.string().min(1),
    architecture: z.string().min(1),
    model_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    config_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    input_size: z.number().int().positive(),
    referable_threshold: finiteProbability,
    calibration_version: z.string().min(1),
    explanation_capability: z.string().min(1),
    build_commit: z.string().min(1),
    deployment_revision: z.string().min(1),
  }),
  dataset: z.string().min(1),
  quality: z.object({
    status: moduleStatus,
    model_version: z.string().min(1),
    score: finiteProbability,
    label: z.enum(['good', 'usable', 'poor']),
    issues: z.array(z.string()),
    brightness: z.number().finite(),
    contrast: z.number().finite(),
    sharpness: z.number().finite(),
  }).passthrough(),
  dr: z.object({
    status: moduleStatus,
    grade: z.number().int().min(0).max(4).nullable(),
    label: z.string(),
    probabilities: z.array(finiteProbability).max(5),
    confidence_raw: finiteProbability.nullable(),
    confidence_calibrated: finiteProbability.nullable(),
    calibration_status: z.enum(['calibrated', 'not_calibrated']),
    referable: z.boolean().nullable(),
    referable_score: finiteProbability.nullable(),
    referable_threshold: finiteProbability,
    referable_decision: z.boolean().nullable(),
  }),
  dme: z.object({ status: moduleStatus, risk: z.number().int().min(0).max(2).nullable(), label: z.string(), probabilities: z.array(finiteProbability) }),
  structures: z.object({
    status: moduleStatus,
    vessels: z.object({ status: moduleStatus }).passthrough(),
    optic_disc: z.object({ status: moduleStatus }).passthrough(),
    fovea: z.object({ status: moduleStatus }).passthrough(),
  }).passthrough(),
  lesions: z.object({ status: moduleStatus, experimental: z.boolean(), items: z.array(z.unknown()) }).passthrough(),
  explainability: z.object({
    status: z.union([moduleStatus, z.literal('baseline_feature_activation')]),
    method: z.string(),
    image: z.string(),
    attention_regions: z.array(z.unknown()),
    clinical_interpretation: z.string(),
  }).passthrough(),
  recommendation: z.object({
    text: z.string().min(1),
    urgency: z.enum(['retake', 'refer', 'review', 'routine']),
    requires_human_review: z.boolean(),
    reasons: z.array(z.string()),
  }),
  uncertainty: z.object({
    label: z.enum(['low', 'medium', 'high', 'unavailable']),
    requires_manual_review: z.boolean(),
    reason: z.string(),
  }),
  runtime: z.object({ processing_mode: z.string(), device: z.string(), latency_ms: z.number().finite().nonnegative() }),
  dr_grade: z.number().int().min(0).max(4).nullable(),
  dme_risk: z.number().int().min(0).max(2).nullable(),
  confidence: finiteProbability.nullable(),
  referable_dr: z.boolean().nullable(),
  disclaimer: z.string().min(1),
}).passthrough().superRefine((value, context) => {
  const notAssessed = value.assessment.state === 'not_assessed' || value.assessment.state === 'retake_required';
  if (notAssessed && (value.dr.grade !== null || value.dr.referable_decision !== null || value.referable_dr !== null)) {
    context.addIssue({ code: 'custom', message: 'Not-assessed results cannot contain a DR grade or referral decision.' });
  }
  if (!notAssessed && value.dr.referable_decision !== value.referable_dr) {
    context.addIssue({ code: 'custom', message: 'Referral compatibility field disagrees with the dedicated referral decision.' });
  }
});

export function persistedStatus(state: AssessmentState): 'completed' | 'needs_retake' | 'failed' {
  if (state === 'retake_required') return 'needs_retake';
  if (state === 'not_assessed') return 'failed';
  return 'completed';
}

export function referralLabel(state: AssessmentState): string {
  if (state === 'retake_required') return 'Retake required';
  if (state === 'not_assessed') return 'Not assessed';
  if (state === 'uncertain') return 'Manual review';
  return state === 'assessed_referable' ? 'Refer' : 'Non-referable';
}
