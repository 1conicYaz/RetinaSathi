import { describe, expect, it } from 'vitest';
import { persistedStatus, referralLabel, screeningResultSchema } from './screening-contract';

function resultFixture() {
  return {
    api_version: '2.0',
    assessment: { state: 'assessed_referable', reason: 'Dedicated referral head crossed its threshold' },
    model_version: 'classifier-v3.4-seed26038',
    model_identity: {
      model_name: 'RetinaSathi', model_version: 'classifier-v3.4-seed26038', architecture: 'dinov2_vits14',
      model_sha256: 'a'.repeat(64), config_sha256: 'b'.repeat(64), input_size: 392,
      referable_threshold: 0.20733, calibration_version: 'manifest-sha256:bbbbbbbbbbbb',
      explanation_capability: 'unavailable_in_onnx_cloud_runtime', build_commit: 'abc123', deployment_revision: 'revision-1',
    },
    dataset: 'APTOS_2019+IDRiD+DeepDRiD',
    quality: { status: 'heuristic', model_version: 'deterministic-quality-v1', score: 0.9, label: 'good', issues: [], brightness: 0.4, contrast: 0.1, sharpness: 0.02 },
    dr: { status: 'candidate', grade: 0, label: 'No DR', probabilities: [0.8, 0.1, 0.05, 0.03, 0.02], confidence_raw: 0.7, confidence_calibrated: 0.8, calibration_status: 'calibrated', referable: true, referable_score: 0.63, referable_threshold: 0.20733, referable_decision: true },
    dme: { status: 'unavailable', risk: null, label: 'Not assessed', probabilities: [] },
    structures: { status: 'unavailable', vessels: { status: 'unavailable' }, optic_disc: { status: 'unavailable' }, fovea: { status: 'unavailable' } },
    lesions: { status: 'unavailable', experimental: true, items: [] },
    explainability: { status: 'unavailable', method: 'unavailable', image: '', attention_regions: [], clinical_interpretation: 'Unavailable' },
    recommendation: { text: 'Refer for review.', urgency: 'refer', requires_human_review: true, reasons: ['Referral head positive'] },
    uncertainty: { label: 'low', requires_manual_review: true, reason: 'Human review required' },
    runtime: { processing_mode: 'cloud', device: 'cpu', latency_ms: 500 },
    dr_grade: 0, dme_risk: null, confidence: 0.8, referable_dr: true, disclaimer: 'Research only',
  };
}

describe('screening result safety contract', () => {
  it('keeps the dedicated referral decision even when grade probabilities look non-referable', () => {
    const parsed = screeningResultSchema.parse(resultFixture());
    expect(parsed.dr.referable_score).toBe(0.63);
    expect(parsed.dr.referable_decision).toBe(true);
    expect(parsed.dr.probabilities.slice(2).reduce((sum, value) => sum + value, 0)).toBeCloseTo(0.1);
  });

  it('rejects a retake result that is serialized as a normal negative', () => {
    const base = resultFixture();
    const fixture = {
      ...base,
      assessment: { state: 'retake_required', reason: 'Poor quality' },
      dr: { ...base.dr, grade: null, referable_decision: false },
      referable_dr: false,
    };
    expect(() => screeningResultSchema.parse(fixture)).toThrow(/Not-assessed/);
  });

  it('preserves distinct history labels for negative, uncertain, and rejected results', () => {
    expect(referralLabel('assessed_non_referable')).toBe('Non-referable');
    expect(referralLabel('uncertain')).toBe('Manual review');
    expect(referralLabel('retake_required')).toBe('Retake required');
    expect(persistedStatus('retake_required')).toBe('needs_retake');
  });
});
