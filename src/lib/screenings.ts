import { getInsforgeClient } from './insforge';

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
  model_version: string;
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
    referable: boolean;
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
  referable_dr: boolean;
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
  created_at: string;
};

export type ReviewDecision = 'reviewed' | 'overridden' | 'inconclusive';

type LegacyScreeningPayload = Omit<ScreeningResult, 'api_version' | 'dr' | 'dme' | 'structures' | 'lesions' | 'explainability' | 'recommendation' | 'uncertainty' | 'runtime' | 'quality' | 'legacy_lesions'> & {
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

function inferenceCandidates(): { mode: 'local' | 'cloud'; url: string }[] {
  const selected = (import.meta.env.VITE_INFERENCE_MODE?.trim() || 'auto') as 'local' | 'cloud' | 'auto';
  const local = (import.meta.env.VITE_LOCAL_INFERENCE_URL?.trim() || 'http://127.0.0.1:8000').replace(/\/$/, '');
  const cloud = import.meta.env.VITE_INFERENCE_URL?.trim()?.replace(/\/$/, '');
  if (selected === 'local') return [{ mode: 'local', url: local }];
  if (selected === 'cloud') {
    if (!cloud) throw new ScreeningError('MODEL_UNAVAILABLE', 'The cloud inference service is not configured.', 'Ask the technical operator to configure VITE_INFERENCE_URL.');
    return [{ mode: 'cloud', url: cloud }];
  }
  return [{ mode: 'local', url: local }, ...(cloud ? [{ mode: 'cloud' as const, url: cloud }] : [])];
}

export function normalizeScreeningResult(payload: ScreeningResult | LegacyScreeningPayload, mode: 'local' | 'cloud'): ScreeningResult {
  if ('api_version' in payload && payload.api_version === '2.0') {
    return { ...payload, runtime: { ...payload.runtime, processing_mode: mode } };
  }
  const legacy = payload as LegacyScreeningPayload;
  const attentionRegions: AttentionRegion[] = (legacy.lesions ?? []).map((region) => ({
    region_type: 'model_attention', strength: region.probability, center_x: region.center_x,
    center_y: region.center_y, radius: region.radius, evidence: 'This region influenced the baseline feature activation.',
  }));
  return {
    ...legacy,
    api_version: '2.0',
    quality: { status: 'heuristic', model_version: 'deterministic-quality-v1', ...legacy.quality },
    dr: { status: 'baseline_v1', grade: legacy.dr_grade, label: legacy.dr_label, probabilities: legacy.grade_probabilities, confidence_raw: legacy.confidence, confidence_calibrated: null, calibration_status: 'not_calibrated', referable: legacy.referable_dr },
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
        const response = await fetch(`${candidate.url}/predict`, { method: 'POST', body: form });
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { detail?: string | ApiErrorDetail } | null;
          const detail = typeof payload?.detail === 'object' ? payload.detail : null;
          const message = detail?.message ?? (typeof payload?.detail === 'string' ? payload.detail : 'The screening service could not analyze this image.');
          const error = new ScreeningError(detail?.code ?? 'INFERENCE_FAILED', message, detail?.next_action ?? 'Retry or contact the technical operator.');
          if (candidate.mode === 'cloud' && [502, 503, 504].includes(response.status) && attempt + 1 < attempts) {
            lastError = error;
            await new Promise((resolve) => window.setTimeout(resolve, 1200 * (attempt + 1)));
            continue;
          }
          if (response.status !== 503 || candidate.mode === 'cloud') throw error;
          lastError = error;
          break;
        }
        const result = (await response.json()) as ScreeningResult | LegacyScreeningPayload;
        return normalizeScreeningResult(result, candidate.mode);
      } catch (caught) {
        if (!(caught instanceof ScreeningError) && candidate.mode === 'cloud' && attempt + 1 < attempts) {
          await new Promise((resolve) => window.setTimeout(resolve, 1200 * (attempt + 1)));
          continue;
        }
        if (caught instanceof ScreeningError && candidate.mode === 'cloud') throw caught;
        lastError = caught instanceof ScreeningError ? caught : new ScreeningError(
          candidate.mode === 'local' ? 'LOCAL_MODEL_UNAVAILABLE' : 'NETWORK_UNAVAILABLE',
          candidate.mode === 'local' ? 'The local model service is unavailable.' : 'The cloud model service cannot be reached after three attempts.',
          candidate.mode === 'local' ? 'Start the local demo service; automatic mode will try cloud next.' : 'Check connectivity or retry in local mode.',
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
): Promise<ScreeningRecord> {
  const insforge = getInsforgeClient();
  const extension = file.name.split('.').pop()?.toLowerCase().replace(/[^a-z0-9]/g, '') || 'jpg';
  const imageKey = `${userId}/${crypto.randomUUID()}.${extension}`;
  const { data: image, error: uploadError } = await insforge.storage
    .from('retinal-screenings')
    .upload(imageKey, file);
  if (uploadError || !image) throw new ScreeningError('STORAGE_FAILED', uploadError?.message ?? 'Could not securely upload the image.', 'The screening can remain in the local sync queue until connectivity returns.');

  const status = result.quality.label === 'poor' ? 'needs_retake' : 'completed';
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

const SCREENING_COLUMNS = 'id, patient_code, patient_age, patient_sex, image_key, status, quality_score, quality_label, dr_grade, dme_risk, confidence, referable_dr, recommendation, model_version, model_confidence_raw, model_confidence_calibrated, uncertainty_label, quality_model_version, classifier_model_version, segmentation_model_version, review_status, reviewed_at, processing_mode, created_at';

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
