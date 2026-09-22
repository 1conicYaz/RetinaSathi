import { useState } from 'react';
import type { PatientDetails, ScreeningResult } from '../lib/screenings';
import { AppIcon } from './app-icon';

const gradeNames = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
const clinicalGradeMeaning = [
  'No visible diabetic-retinopathy signs are expected.',
  'Microaneurysms only.',
  'More than microaneurysms, but fewer findings than severe NPDR.',
  'The clinical reference is the 4-2-1 rule: extensive retinal haemorrhages, venous beading, or prominent IRMA, without proliferative signs.',
  'Neovascularisation or vitreous/preretinal haemorrhage may be present.',
];

export function ResultPanel({ result, originalUrl, patient }: { result: ScreeningResult; originalUrl: string; patient: PatientDetails }) {
  const [view, setView] = useState<'operator' | 'reviewer'>('operator');
  const urgent = result.recommendation.urgency !== 'routine';
  const gradeLabel = result.dr_grade == null ? 'Not graded' : `${result.dr_grade} · ${gradeNames[result.dr_grade]}`;
  const confidenceLabel = result.confidence == null ? 'Unavailable' : `${Math.round(result.confidence * 100)}%`;
  const dmeLabel = result.dme_risk == null ? 'Not assessed' : `${result.dme_risk} of 2`;
  const isGradCam = result.explainability.method.startsWith('gradcam');
  const explanationAvailable = result.explainability.status !== 'unavailable';
  const explanationLabel = isGradCam ? 'Grad-CAM' : explanationAvailable ? 'Feature activation' : 'Attention map unavailable';
  const modelShortName = result.model_version.startsWith('classifier-v3.4') ? 'V3.4' : result.model_version.startsWith('classifier-v2') ? 'V2' : 'V1';
  const referableProbability = result.dr.probabilities.slice(2).reduce((total, probability) => total + probability, 0);
  const strongestAttention = result.explainability.attention_regions.reduce((maximum, region) => Math.max(maximum, region.strength), 0);
  const selectedGradeProbability = result.dr_grade == null ? null : result.dr.probabilities[result.dr_grade] ?? result.confidence;
  const probabilityKind = result.dr.calibration_status === 'calibrated' ? 'calibrated probability' : 'model probability';
  const competingGrades = result.dr.probabilities
    .map((probability, grade) => ({ grade, probability }))
    .filter(({ grade }) => grade !== result.dr_grade)
    .sort((left, right) => right.probability - left.probability)
    .slice(0, 2);
  const hindiAction = result.quality.label === 'poor'
    ? 'छवि दोबारा लें। इस छवि से DR या DME परिणाम नहीं बनाया गया।'
    : result.recommendation.urgency === 'review'
      ? 'कम भरोसे वाले परिणाम की चिकित्सक से समीक्षा कराएं या छवि दोबारा लें।'
      : result.referable_dr
      ? 'पुष्टि के लिए नेत्र विशेषज्ञ को रेफर करें।'
      : 'नियमित फॉलो-अप स्क्रीनिंग करें।';
  return (
    <section className="result-card" aria-live="polite">
      <div className="result-heading">
        <div>
          <p className="eyebrow">Screening result</p>
          <h2>{result.quality.label === 'poor' ? 'Image retake required' : result.dr_label}</h2>
        </div>
        <span className={`urgency-pill ${urgent ? 'urgency-pill--urgent' : ''}`}>
          {result.recommendation.urgency === 'review' ? 'Review needed' : urgent ? 'Action needed' : 'Routine follow-up'}
        </span>
      </div>

      <div className="view-toggle" role="group" aria-label="Result detail level"><button type="button" className={view === 'operator' ? 'view-toggle--active' : ''} onClick={() => setView('operator')}>Operator view</button><button type="button" className={view === 'reviewer' ? 'view-toggle--active' : ''} onClick={() => setView('reviewer')}>Reviewer details</button></div>

      <dl className={`report-meta advanced-only ${view === 'operator' ? 'advanced-only--hidden' : ''}`}>
        <div><dt>Patient reference</dt><dd>{patient.code}</dd></div>
        <div><dt>Age / sex</dt><dd>{patient.age ?? 'Not recorded'} · {patient.sex}</dd></div>
        <div><dt>Generated</dt><dd>{new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date())}</dd></div>
        <div><dt>Model</dt><dd>{result.model_version}</dd></div>
        <div><dt>Review status</dt><dd>Pending human review</dd></div>
        <div><dt>Calibration</dt><dd>{result.dr.calibration_status.replaceAll('_', ' ')}</dd></div>
      </dl>

      <div className="explanation-summary">
        <span>{explanationLabel}</span>
        <p>{!explanationAvailable
          ? 'The cloud ONNX model provides calibrated grading and referral output, but it does not contain the gradients needed for an attention map. Use the local V3.4 runtime for the experimental explanation.'
          : isGradCam
          ? 'Brighter colours show image areas that influenced the model more. This is model attention, not a confirmed lesion.'
          : 'This is a coarse feature-activation map. It is not Grad-CAM, lesion detection, or anatomical proof.'}</p>
      </div>

      <div className={`image-compare ${result.explainability.heatmap_image ? 'image-compare--three' : ''}`}>
        <figure><img src={originalUrl} alt="Uploaded retinal photograph"/><figcaption>Original fundus image</figcaption></figure>
        {result.explainability.heatmap_image ? <figure className="heatmap-figure"><img src={result.explainability.heatmap_image} alt={`${explanationLabel} standalone colour heatmap`}/><figcaption>Attention-only heatmap<div className="heatmap-legend" aria-label="Attention strength from lower to higher"><span>Lower</span><i/><span>Higher</span></div></figcaption></figure> : null}
        {explanationAvailable ? <figure><img src={result.explanation_image} alt="Model attention heatmap overlaid on the retinal image"/><figcaption>{explanationLabel} overlay</figcaption></figure> : null}
      </div>

      <div className="result-stats">
        <article><span>DR grade</span><strong>{gradeLabel}</strong><small>{result.dr_grade == null ? 'Quality gate stopped inference' : 'Five-level IDRiD scale'}</small></article>
        <article><span>Confidence</span><strong>{confidenceLabel}</strong><small>{result.confidence == null ? 'No model prediction' : 'Model probability'}</small></article>
        <article><span>DME risk</span><strong>{dmeLabel}</strong><small>{result.dme_label}</small></article>
        <article><span>Image quality</span><strong>{Math.round(result.quality.score * 100)}%</strong><small className={`quality-${result.quality.label}`}>{result.quality.label}</small></article>
      </div>

      {result.dr_grade != null ? <section className="decision-explanation" aria-labelledby="decision-explanation-title">
        <div className="section-title"><div><p className="eyebrow">Why this result?</p><h3 id="decision-explanation-title">How {modelShortName} reached “Grade {result.dr_grade} · {gradeNames[result.dr_grade]}”</h3></div></div>
        <ol>
          <li><strong>Grade choice</strong><p>Grade {result.dr_grade} had the highest {probabilityKind} at {Math.round((selectedGradeProbability ?? 0) * 100)}%. The nearest alternatives were {competingGrades.map(({ grade, probability }) => `Grade ${grade} at ${Math.round(probability * 100)}%`).join(' and ')}.</p></li>
          <li><strong>Referral decision</strong><p>The combined probability of referable DR (Grades 2–4) was {Math.round(referableProbability * 100)}%. {result.referable_dr ? 'This crossed the model’s referral operating point, so ophthalmologist review is recommended.' : 'This remained below the model’s referral operating point, but routine follow-up and human review still apply.'}</p></li>
          <li><strong>Image and explanation evidence</strong><p>The image passed the quality check at {Math.round(result.quality.score * 100)}%. {explanationAvailable ? `The explanation found ${result.explainability.attention_regions.length} influential region${result.explainability.attention_regions.length === 1 ? '' : 's'}${strongestAttention ? `, with the strongest attention score at ${Math.round(strongestAttention * 100)}%` : ''}. These regions influenced the score but are not confirmed lesions.` : 'This cloud export does not provide an attention map. The grade is explained by its calibrated probabilities and referral score; the local V3.4 runtime can produce an experimental attention view.'}</p></li>
        </ol>
        <div className="clinical-meaning"><strong>Clinical meaning of this grade — reference only</strong><p>{clinicalGradeMeaning[result.dr_grade]}</p><p>{modelShortName} has not verified the required lesions because lesion segmentation is not validated. An ophthalmologist must compare the original image with the clinical grading criteria before confirming the grade.</p></div>
        <p className="decision-explanation__hindi" lang="hi">मॉडल ने ग्रेड {result.dr_grade} चुना क्योंकि इसी ग्रेड की संभावना सबसे अधिक थी। यह किसी विशेष घाव की पुष्टि नहीं है। अंतिम पुष्टि नेत्र विशेषज्ञ करेंगे।</p>
      </section> : null}

      <div className={`recommendation ${urgent ? 'recommendation--urgent' : ''}`}>
        <AppIcon name={urgent ? 'alert' : 'check'} size={22}/>
        <div><strong>Recommended next step</strong><p>{result.recommendation.text}</p><p lang="hi">{hindiAction}</p></div>
      </div>

      {result.quality.issues.length ? <div className="quality-issues"><strong>Capture feedback</strong><ul>{result.quality.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul></div> : null}

      <div className={`evidence-list advanced-only ${view === 'operator' ? 'advanced-only--hidden' : ''}`}>
        <div className="section-title"><div><p className="eyebrow">Explainability</p><h3>Regions that influenced the result</h3></div><span>{result.explainability.attention_regions.length} regions</span></div>
        <p className="evidence-warning">{!explanationAvailable ? result.explainability.clinical_interpretation : result.explainability.method.startsWith('gradcam') ? 'Grad-CAM shows classifier influence — it is not a lesion map or anatomical confirmation.' : 'Feature activation only — not Grad-CAM, a lesion map, or anatomical confirmation.'}</p>
        {result.explainability.attention_regions.length ? result.explainability.attention_regions.map((region, index) => (
          <div className="evidence-row" key={`${region.center_x}-${region.center_y}`}>
            <span className="evidence-number">{String(index + 1).padStart(2, '0')}</span>
            <div><strong>Model attention</strong><p>{region.evidence}</p></div>
            <span>{Math.round(region.strength * 100)}%</span>
          </div>
        )) : <p className="muted-copy">{explanationAvailable ? 'No strong focal attention regions were found.' : 'No cloud attention map was generated. Use local V3.4 when an experimental explanation is needed.'}</p>}
      </div>

      {(result.lesions.overlay_image || result.structures.vessels.overlay_image || result.structures.localization_overlay_image) ? <div className={`advanced-only ${view === 'operator' ? 'advanced-only--hidden' : ''}`}>
        <div className="section-title"><div><p className="eyebrow">Experimental retinal analysis</p><h3>Separate structure and lesion overlays</h3></div></div>
        <p className="evidence-warning">These mask-trained research outputs are separate from Grad-CAM and require clinician review.</p>
        <div className="image-compare">
          {result.lesions.overlay_image ? <figure><img src={result.lesions.overlay_image} alt="Experimental lesion segmentation overlay"/><figcaption>Four-class lesion overlay · processed square frame</figcaption></figure> : null}
          {result.structures.vessels.overlay_image ? <figure><img src={result.structures.vessels.overlay_image} alt="Experimental retinal vessel overlay"/><figcaption>Vessel overlay · processed square frame</figcaption></figure> : null}
          {result.structures.localization_overlay_image ? <figure><img src={result.structures.localization_overlay_image} alt="Experimental optic disc and fovea localization"/><figcaption>Disc and fovea localization</figcaption></figure> : null}
        </div>
        {result.lesions.items.length ? <div className="module-statuses"><h3>Lesion-mask summary</h3>{result.lesions.items.map((item) => <div key={item.type}><span>{item.type.replaceAll('_', ' ')}</span><strong>{(item.pixel_fraction * 100).toFixed(3)}% of processed image</strong></div>)}</div> : null}
      </div> : null}

      <div className={`module-statuses advanced-only ${view === 'operator' ? 'advanced-only--hidden' : ''}`} aria-label="Retinal analysis module readiness">
        <h3>Analysis module status</h3>
        <div><span>DR classifier</span><strong>{result.dr.status.replaceAll('_', ' ')}</strong></div>
        <div><span>Quality assessment</span><strong>{result.quality.status}</strong></div>
        <div><span>Lesion segmentation</span><strong>{result.lesions.status.replaceAll('_', ' ')}</strong></div>
        <div><span>Vessels</span><strong>{result.structures.vessels.status.replaceAll('_', ' ')}</strong></div>
        <div><span>Optic disc</span><strong>{result.structures.optic_disc.status.replaceAll('_', ' ')}{result.structures.optic_disc.confidence != null ? ` · ${Math.round(result.structures.optic_disc.confidence * 100)}% relative confidence` : ''}</strong></div>
        <div><span>Fovea</span><strong>{result.structures.fovea.status.replaceAll('_', ' ')}{result.structures.fovea.confidence != null ? ` · ${Math.round(result.structures.fovea.confidence * 100)}% relative confidence` : ''}</strong></div>
        <div><span>Confidence calibration</span><strong>{result.dr.calibration_status.replaceAll('_', ' ')}</strong></div>
        <div><span>Uncertainty routing</span><strong>{result.uncertainty.label} · {result.uncertainty.reason}</strong></div>
        <div><span>Runtime</span><strong>{result.runtime.processing_mode} · {Math.round(result.runtime.latency_ms)} ms</strong></div>
        <div><span>Grade probabilities</span><strong>{result.dr.probabilities.length ? result.dr.probabilities.map((value, grade) => `G${grade} ${Math.round(value * 100)}%`).join(' · ') : 'Not available'}</strong></div>
      </div>

      <div className="result-footer">
        <p>{result.disclaimer} Human review is required.</p>
        <button type="button" className="button button--secondary" onClick={() => window.print()}>Print / save PDF</button>
      </div>
    </section>
  );
}
