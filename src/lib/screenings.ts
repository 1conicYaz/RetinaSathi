import { getInsforgeClient } from './insforge';
import { ZodError } from 'zod';
import { persistedStatus, screeningResultSchema, type AssessmentState } from './screening-contract';

export type PatientSex = 'female' | 'male' | 'other' | 'unknown';

export type ModuleStatus = 'ready' | 'candidate' | 'baseline_v1' | 'heuristic' | 'not_trained' | 'data_limited' | 'unavailable';

export type AttentionRegion = {
  region_type: 'model_attention';
  strength: number;
  center_x: number;
  center_y: number;
  radius: number;
  evidence: string;
};

export type ScreeningResult = {
  api_version: '2.0';
  assessment: { state: AssessmentState; reason: string };
  model_version: string;
  model_identity: {
    model_name: string;
    model_version: string;
    architecture: string;
    model_sha256: string;
    config_sha256: string;
    input_size: number;
    referable_threshold: number;
    calibration_version: string;
    explanation_capability: string;
    build_commit: string;
    deployment_revision: string;
  };
  dataset: string;
  quality: {
    status: ModuleStatus;
    model_version: string;
    score: number;
    label: 'good' | 'usable' | 'poor';
    issues: string[];
    brightness: number;
    contrast: number;
    sharpness: number;
    learned_good_probability?: number;
    learned_attributes?: { artifact: number; clarity: number; field_definition: number };
  };
  dr: {
    status: ModuleStatus;
    grade: number | null;
    label: string;
    probabilities: number[];
    confidence_raw: number | null;
    confidence_calibrated: number | null;
    calibration_status: 'calibrated' | 'not_calibrated';
    referable: boolean | null;
    referable_score: number | null;
    referable_threshold: number;
    referable_decision: boolean | null;
  };
  dme: { status: ModuleStatus; risk: number | null; label: string; probabilities: number[] };
  structures: {
    status: ModuleStatus;
    vessels: { status: ModuleStatus; model_version?: string; pixel_fraction?: number; overlay_image?: string };
    optic_disc: { status: ModuleStatus; model_version?: string; x?: number; y?: number; confidence?: number };
    fovea: { status: ModuleStatus; model_version?: string; x?: number; y?: number; confidence?: number };
    localization_overlay_image?: string;
  };
  lesions: {
    status: ModuleStatus;
    experimental: boolean;
    model_version?: string | null;
    overlay_image?: string | null;
    items: Array<{ type: string; status: 'experimental'; pixel_fraction: number; mean_probability_above_threshold: number | null }>;
  };
  explainability: {
    status: ModuleStatus | 'baseline_feature_activation';
    method: string;
    image: string;
    heatmap_image?: string;
    attention_regions: AttentionRegion[];
    clinical_interpretation: string;
  };
  recommendation: {
    text: string;
    urgency: 'retake' | 'refer' | 'review' | 'routine';
    requires_human_review: boolean;
    reasons: string[];
  };
  uncertainty: { label: 'low' | 'medium' | 'high' | 'unavailable'; requires_manual_review: boolean; reason: string };
  runtime: { processing_mode: string; device: string; latency_ms: number };
  dr_grade: number | null;
  dr_label: string;
  grade_probabilities: number[];
  dme_risk: number | null;
  dme_label: string;
  dme_probabilities: number[];
  confidence: number | null;
  referable_dr: boolean | null;
  recommendation_text: string;
  explanation_image: string;
  legacy_lesions: [];
  disclaimer: string;
};

export type ScreeningRecord = {
  id: string;
  patient_code: string;
  patient_age: number | null;
  patient_sex: PatientSex | null;
  image_key: string;
  status: 'queued' | 'processing' | 'completed' | 'needs_retake' | 'failed';
  quality_score: number | null;
  quality_label: 'good' | 'usable' | 'poor' | null;
  dr_grade: number | null;
  dme_risk: number | null;
  confidence: number | null;
  referable_dr: boolean | null;
  recommendation: string | null;
  model_version: string;
  model_confidence_raw: number | null;
  model_confidence_calibrated: number | null;
  uncertainty_label: 'low' | 'medium' | 'high' | 'unavailable' | null;
  quality_model_version: string | null;
  classifier_model_version: string | null;
  segmentation_model_version: string | null;
  review_status: 'pending' | 'reviewed' | 'overridden' | 'inconclusive';
  reviewed_at: string | null;
  processing_mode: 'local' | 'cloud' | 'auto';
  assessment_state: AssessmentState;
  referral_score: number | null;
  referral_threshold: number | null;
  referral_decision: boolean | null;
  model_sha256: string | null;
  config_sha256: string | null;
  calibration_version: string | null;
  explanation_status: string | null;
  operation_id: string | null;
  result_snapshot: Record<string, unknown> | null;
  created_at: string;
};

export type ReviewDecision = 'reviewed' | 'overridden' | 'inconclusive';

type LegacyScreeningPayload = Omit<ScreeningResult, 'api_version' | 'assessment' | 'model_identity' | 'dr' | 'dme' | 'structures' | 'lesions' | 'explainability' | 'recommendation' | 'uncertainty' | 'runtime' | 'quality' | 'legacy_lesions'> & {
  quality: Omit<ScreeningResult['quality'], 'status' | 'model_version'>;
  recommendation: string;
  lesions?: Array<{ probability: number; center_x: number; center_y: number; radius: number; evidence: string }>;
};

export type PatientDetails = {
  code: string;
  age: number | null;
  sex: PatientSex;
};

type ApiErrorDetail = { code?: string; message?: string; next_action?: string };

export class ScreeningError extends Error {
  code: string;
  nextAction: string;

  constructor(code: string, message: string, nextAction: string) {
    super(message);
    this.code = code;
    this.nextAction = nextAction;
  }
}

type InferenceCandidate =
  | { mode: 'local'; url: string }
  | { mode: 'cloud'; functionSlug: string };

function inferenceCandidates(): InferenceCandidate[] {
  const selected = (import.meta.env.VITE_INFERENCE_MODE?.trim() || 'auto') as 'local' | 'cloud' | 'auto';
  const local = (import.meta.env.VITE_LOCAL_INFERENCE_URL?.trim() || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const cloudFunction = import.meta.env.VITE_INFERENCE_FUNCTION?.trim() || 'retinasathi-inference';
  if (selected === 'local') return [{ mode: 'local', url: local }];
  if (selected === 'cloud') return [{ mode: 'cloud', functionSlug: cloudFunction }];
  return [{ mode: 'local', url: local }, { mode: 'cloud', functionSlug: cloudFunction }];
}

export function normalizeScreeningResult(payload: ScreeningResult | LegacyScreeningPayload, mode: 'local' | 'cloud'): ScreeningResult {
  if ('api_version' in payload && payload.api_version === '2.0') {
    try {
      return screeningResultSchema.parse({ ...payload, runtime: { ...payload.runtime, processing_mode: mode } }) as ScreeningResult;
    } catch (error) {
      if (error instanceof ZodError) {
        throw new ScreeningError('INVALID_MODEL_RESPONSE', 'The model returned an incompatible result contract.', 'Do not use this result; contact the technical operator.');
      }
      throw error;
    }
  }
  const legacy = payload as LegacyScreeningPayload;
  const attentionRegions: AttentionRegion[] = (legacy.lesions ?? []).map((region) => ({
    region_type: 'model_attention', strength: region.probability, center_x: region.center_x,
    center_y: region.center_y, radius: region.radius, evidence: 'This region influenced the baseline feature activation.',
  }));
  return {
    ...legacy,
    api_version: '2.0',
    assessment: {
      state: legacy.quality.label === 'poor' ? 'retake_required' : legacy.referable_dr ? 'assessed_referable' : 'uncertain',
      reason: legacy.quality.label === 'poor' ? 'Image quality gate stopped inference' : 'Legacy V1 result requires manual review',
    },
    model_identity: {
      model_name: 'RetinaSathi legacy baseline', model_version: legacy.model_version,
      architecture: 'MobileNetV3', model_sha256: '0'.repeat(64), config_sha256: '0'.repeat(64),
      input_size: 224, referable_threshold: 0.5, calibration_version: 'unavailable',
      explanation_capability: 'baseline_feature_activation', build_commit: 'unavailable', deployment_revision: 'unavailable',
    },
    quality: { status: 'heuristic', model_version: 'deterministic-quality-v1', ...legacy.quality },
    dr: { status: 'baseline_v1', grade: legacy.dr_grade, label: legacy.dr_label, probabilities: legacy.grade_probabilities, confidence_raw: legacy.confidence, confidence_calibrated: null, calibration_status: 'not_calibrated', referable: legacy.quality.label === 'poor' ? null : legacy.referable_dr, referable_score: null, referable_threshold: 0.5, referable_decision: legacy.quality.label === 'poor' ? null : legacy.referable_dr },
    dme: { status: 'baseline_v1', risk: legacy.dme_risk, label: legacy.dme_label, probabilities: legacy.dme_probabilities },
    structures: { status: 'not_trained', vessels: { status: 'not_trained' }, optic_disc: { status: 'not_trained' }, fovea: { status: 'not_trained' } },
    lesions: { status: 'not_trained', experimental: true, items: [] },
    explainability: { status: 'baseline_feature_activation', method: 'channel_mean_feature_activation_not_gradcam', image: legacy.explanation_image, attention_regions: attentionRegions, clinical_interpretation: 'Not a lesion map and not anatomical confirmation.' },
    recommendation: {
      text: legacy.quality.label === 'poor'
        ? legacy.recommendation
        : legacy.referable_dr
          ? legacy.recommendation
          : 'V1 confidence is not calibrated. Obtain qualified human review before deciding routine follow-up.',
      urgency: legacy.quality.label === 'poor' ? 'retake' : legacy.referable_dr ? 'refer' : 'review',
      requires_human_review: true,
      reasons: ['V1 confidence is uncalibrated'],
    },
    uncertainty: { label: 'unavailable', requires_manual_review: true, reason: 'V1 confidence is not calibrated' },
    runtime: { processing_mode: mode, device: 'cloud_cpu_unknown', latency_ms: 0 },
    referable_dr: legacy.quality.label === 'poor' ? null : legacy.referable_dr,
    recommendation_text: legacy.recommendation,
    legacy_lesions: [],
  };
}

export async function analyzeRetina(file: File): Promise<ScreeningResult> {
  const candidates = inferenceCandidates();
  let lastError: ScreeningError | null = null;
  for (const candidate of candidates) {
    const attempts = candidate.mode === 'cloud' ? 3 : 1;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const form = new FormData();
      form.append('file', file);
      try {
        if (candidate.mode === 'cloud') {
          const { data, error } = await getInsforgeClient().functions.invoke<ScreeningResult | LegacyScreeningPayload>(
            candidate.functionSlug,
            { body: form },
          );
          if (error || !data) {
            const functionError = error as (typeof error & {
              statusCode?: number;
              error?: string;
              next_action?: string;
            }) | null;
            const status = functionError?.statusCode ?? 500;
            const mapped = new ScreeningError(
              functionError?.error ?? 'INFERENCE_FAILED',
              functionError?.message ?? 'The cloud screening service could not analyze this image.',
              functionError?.next_action ?? 'Retry or contact the technical operator.',
            );
            if ([502, 503, 504].includes(status) && attempt + 1 < attempts) {
              lastError = mapped;
              await new Promise((resolve) => window.setTimeout(resolve, 1200 * (attempt + 1)));
              continue;
            }
            throw mapped;
          }
          return normalizeScreeningResult(data, candidate.mode);
        }
        const response = await fetch(`${candidate.url}/predict`, { method: 'POST', body: form });
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { detail?: string | ApiErrorDetail } | null;
          const detail = typeof payload?.detail === 'object' ? payload.detail : null;
          const message = detail?.message ?? (typeof payload?.detail === 'string' ? payload.detail : 'The screening service could not analyze this image.');
          const error = new ScreeningError(detail?.code ?? 'INFERENCE_FAILED', message, detail?.next_action ?? 'Retry or contact the technical operator.');
          if (response.status !== 503) throw error;
          lastError = error;
          break;
        }
        const result = (await response.json()) as ScreeningResult | LegacyScreeningPayload;
        return normalizeScreeningResult(result, candidate.mode);
      } catch (caught) {
        lastError = caught instanceof ScreeningError ? caught : new ScreeningError(
          'LOCAL_MODEL_UNAVAILABLE',
          'The local model service is unavailable.',
          'Start the local demo service; automatic mode will try cloud next.',
        );
        break;
      }
    }
  }
  throw lastError ?? new ScreeningError('MODEL_UNAVAILABLE', 'No inference runtime is configured.', 'Configure a local or cloud runtime.');
}

export async function saveScreening(
  userId: string,
  patient: PatientDetails,
  file: File,
  result: ScreeningResult,
  operationId: string = crypto.randomUUID(),
): Promise<ScreeningRecord> {
  const insforge = getInsforgeClient();
  const { data: existingRows, error: lookupError } = await insforge.database
    .from('screenings')
    .select(SCREENING_COLUMNS)
    .eq('operation_id', operationId)
    .limit(1);
  if (lookupError) throw new ScreeningError('STORAGE_FAILED', lookupError.message ?? 'Could not reconcile this screening.', 'Keep the queued copy and retry when connectivity is stable.');
  const existing = (existingRows as ScreeningRecord[] | null)?.[0];
  if (existing) return existing;
  const extension = file.name.split('.').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '') || 'jpg';
  const imageKey = `${userId}/${operationId}.${extension}`;
  const { data: image, error: uploadError } = await insforge.storage
    .from('retinal-screenings')
    .upload(imageKey, file);
  if (uploadError || !image) throw new ScreeningError('STORAGE_FAILED', uploadError?.message ?? 'Could not securely upload the image.', 'The screening can remain in the local sync queue until connectivity returns.');

  const status = persistedStatus(result.assessment.state);
  const uncertainty = result.uncertainty.label;
  const processingMode = ['local', 'cloud', 'auto'].includes(result.runtime.processing_mode)
    ? result.runtime.processing_mode as 'local' | 'cloud' | 'auto'
    : 'auto';
  const { data: rows, error: insertError } = await insforge.database
    .from('screenings')
    .insert([
      {
        user_id: userId,
        patient_code: patient.code,
        patient_age: patient.age,
        patient_sex: patient.sex,
        image_url: image.url,
        image_key: image.key,
        status,
        quality_score: result.quality.score,
        quality_label: result.quality.label,
        dr_grade: result.dr_grade,
        dme_risk: result.dme_risk,
        confidence: result.confidence,
        referable_dr: result.referable_dr,
        recommendation: result.recommendation.text,
        model_version: result.model_version,
        model_confidence_raw: result.dr.confidence_raw,
        model_confidence_calibrated: result.dr.confidence_calibrated,
        uncertainty_label: uncertainty,
        quality_model_version: result.quality.model_version,
        classifier_model_version: result.model_version,
        segmentation_model_version: result.lesions.status === 'ready' ? result.lesions.model_version ?? null : null,
        review_status: 'pending',
        processing_mode: processingMode,
        assessment_state: result.assessment.state,
        referral_score: result.dr.referable_score,
        referral_threshold: result.dr.referable_threshold,
        referral_decision: result.dr.referable_decision,
        model_sha256: result.model_identity.model_sha256,
        config_sha256: result.model_identity.config_sha256,
        calibration_version: result.model_identity.calibration_version,
        explanation_status: result.explainability.status,
        operation_id: operationId,
        result_snapshot: buildResultSnapshot(result),
      },
    ])
    .select(SCREENING_COLUMNS);

  const screening = (rows as ScreeningRecord[] | null)?.[0];
  if (insertError || !screening) {
    await insforge.storage.from('retinal-screenings').remove(image.key);
    throw new ScreeningError('STORAGE_FAILED', insertError?.message ?? 'Could not save the screening record.', 'The screening can remain in the local sync queue until connectivity returns.');
  }

  return screening;
}

export async function listScreenings(): Promise<ScreeningRecord[]> {
  const { data, error } = await getInsforgeClient().database
    .from('screenings')
    .select(SCREENING_COLUMNS)
    .order('created_at', { ascending: false })
    .limit(50);
  if (error) throw new Error(error.message ?? 'Could not load screening history.');
  return (data ?? []) as ScreeningRecord[];
}

const SCREENING_COLUMNS = 'id, patient_code, patient_age, patient_sex, image_key, status, quality_score, quality_label, dr_grade, dme_risk, confidence, referable_dr, recommendation, model_version, model_confidence_raw, model_confidence_calibrated, uncertainty_label, quality_model_version, classifier_model_version, segmentation_model_version, review_status, reviewed_at, processing_mode, assessment_state, referral_score, referral_threshold, referral_decision, model_sha256, config_sha256, calibration_version, explanation_status, operation_id, result_snapshot, created_at';

function buildResultSnapshot(result: ScreeningResult): Record<string, unknown> {
  return {
    api_version: result.api_version,
    assessment: result.assessment,
    model_version: result.model_version,
    model_identity: result.model_identity,
    quality: result.quality,
    dr: result.dr,
    dme: result.dme,
    structures: {
      status: result.structures.status,
      vessels: { status: result.structures.vessels.status },
      optic_disc: { status: result.structures.optic_disc.status },
      fovea: { status: result.structures.fovea.status },
    },
    lesions: { status: result.lesions.status, experimental: result.lesions.experimental },
    explainability: {
      status: result.explainability.status,
      method: result.explainability.method,
      clinical_interpretation: result.explainability.clinical_interpretation,
    },
    recommendation: result.recommendation,
    uncertainty: result.uncertainty,
    runtime: result.runtime,
    disclaimer: result.disclaimer,
  };
}

export async function submitScreeningReview(
  userId: string,
  screeningId: string,
  decision: ReviewDecision,
  reviewerGrade: number | null,
  notes: string,
): Promise<void> {
  const { error } = await getInsforgeClient().database.from('screening_reviews').insert([{
    screening_id: screeningId,
    owner_id: userId,
    reviewer_id: userId,
    decision,
    reviewer_grade: reviewerGrade,
    notes: notes.trim() || null,
  }]);
  if (error) throw new Error(error.message ?? 'Could not save the human review.');
}

export async function deleteScreening(record: ScreeningRecord): Promise<void> {
  const insforge = getInsforgeClient();
  const { error: storageError } = await insforge.storage.from('retinal-screenings').remove(record.image_key);
  if (storageError) throw new Error(storageError.message ?? 'Could not remove the stored image.');
  const { error } = await insforge.database.from('screenings').delete().eq('id', record.id);
  if (error) throw new Error(error.message ?? 'Could not delete the screening record.');
}
