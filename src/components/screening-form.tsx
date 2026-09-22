import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../lib/auth-context';
import { analyzeRetina, saveScreening, ScreeningError, type PatientDetails, type PatientSex, type ScreeningResult } from '../lib/screenings';
import { AppIcon } from './app-icon';
import { ResultPanel } from './result-panel';
import { clearPendingScreenings, pendingScreenings, queueScreening, syncPendingScreenings } from '../lib/offline-queue';

const MAX_BYTES = 15 * 1024 * 1024;

export function ScreeningForm() {
  const localModel = import.meta.env.VITE_INFERENCE_MODE?.trim() === 'local';
  const localModelLabel = import.meta.env.VITE_LOCAL_MODEL_LABEL?.trim() || 'Local research model';
  const { viewer } = useAuth();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [patientCode, setPatientCode] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState<PatientSex>('unknown');
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [stage, setStage] = useState<'idle' | 'quality' | 'grading' | 'explaining' | 'saving'>('idle');
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');
  const [pendingCount, setPendingCount] = useState(0);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    if (!viewer.id) return;
    const userId = viewer.id;
    void pendingScreenings(userId).then((rows) => setPendingCount(rows.length)).catch(() => setSyncMessage('This browser could not inspect its pending screening queue.'));
    const synchronize = () => {
      if (!navigator.onLine) return;
      void syncPendingScreenings(userId, saveScreening).then(({ synchronized, remaining }) => {
        setPendingCount(remaining);
        if (synchronized) setSyncMessage(`${synchronized} locally queued screening${synchronized === 1 ? '' : 's'} synchronized securely.`);
        else if (remaining) setSyncMessage(`${remaining} screening${remaining === 1 ? '' : 's'} still waiting to synchronize.`);
      }).catch(() => setSyncMessage('Pending screenings could not be synchronized. Retry when connectivity is stable.'));
    };
    window.addEventListener('online', synchronize);
    synchronize();
    return () => window.removeEventListener('online', synchronize);
  }, [viewer.id]);

  function chooseFile(next: File | undefined) {
    setError('');
    setResult(null);
    setSaved(false);
    setSyncMessage('');
    if (!next) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(next.type)) {
      setError('Choose a JPEG, PNG, or WebP retinal image.');
      return;
    }
    if (next.size > MAX_BYTES) {
      setError('The image must be smaller than 15 MB.');
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next);
    setPreview(URL.createObjectURL(next));
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !viewer.id || !patientCode.trim()) return;
    setError('');
    setResult(null);
    setSaved(false);
    try {
      setStage('quality');
      const stageTimer = window.setTimeout(() => setStage('grading'), 700);
      const explanationTimer = window.setTimeout(() => setStage('explaining'), 1700);
      const prediction = await analyzeRetina(file);
      window.clearTimeout(stageTimer);
      window.clearTimeout(explanationTimer);
      setResult(prediction);
      setStage('saving');
      const patient: PatientDetails = {
        code: patientCode.trim(),
        age: age ? Number(age) : null,
        sex,
      };
      try {
        await saveScreening(viewer.id, patient, file, prediction);
        setSaved(true);
      } catch (saveError) {
        if (!(saveError instanceof ScreeningError) || saveError.code !== 'STORAGE_FAILED') throw saveError;
        await queueScreening(viewer.id, patient, file, prediction);
        setPendingCount((count) => count + 1);
        setSyncMessage('SYNC PENDING: saved only in this browser. Keep this device secure and reconnect to upload it.');
      }
    } catch (caught) {
      setError(caught instanceof ScreeningError ? `${caught.message} ${caught.nextAction}` : caught instanceof Error ? caught.message : 'Screening failed. Please try again.');
    } finally {
      setStage('idle');
    }
  }

  const busy = stage !== 'idle';
  const stageLabel = stage === 'quality' ? 'Checking image quality…' : stage === 'grading' ? 'Grading diabetic retinopathy…' : stage === 'explaining' ? 'Building visual explanation…' : 'Saving securely to InsForge…';

  return (
    <div className="screening-layout">
      <form className="screening-form" onSubmit={handleSubmit}>
        <div className="panel-heading"><div><p className="eyebrow">New screening</p><h1>Analyze a retinal image</h1></div><span className="step-chip">{localModel ? localModelLabel : 'Cloud model'}</span></div>
        <div className="privacy-note"><AppIcon name="shield" size={18}/><p><strong>Private by design.</strong> The image is stored in your protected InsForge workspace after analysis.<br/><span lang="hi">मरीज के नाम की जगह केवल पहचान कोड लिखें। साफ और पूरी रेटिना की तस्वीर लें।</span></p></div>

        <div className="form-grid">
          <label className="field"><span>Patient reference</span><input required maxLength={32} value={patientCode} onChange={(event) => setPatientCode(event.target.value)} placeholder="Example: PHC-0042"/><small>Use a code, not the patient’s name.</small></label>
          <label className="field"><span>Age</span><input type="number" min="1" max="120" value={age} onChange={(event) => setAge(event.target.value)} placeholder="54"/></label>
          <label className="field"><span>Sex</span><select value={sex} onChange={(event) => setSex(event.target.value as PatientSex)}><option value="unknown">Prefer not to say</option><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option></select></label>
        </div>

        <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => chooseFile(event.target.files?.[0])}/>
        <button type="button" className={`upload-zone ${preview ? 'upload-zone--filled' : ''}`} onClick={() => inputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}>
          {preview ? <><img src={preview} alt="Selected retinal preview"/><span className="replace-image">Choose another image</span></> : <><span className="upload-icon"><AppIcon name="upload" size={26}/></span><strong>Drop a fundus image here</strong><span>or tap to browse · JPEG, PNG or WebP · max 15 MB</span></>}
        </button>

        {error ? <div className="form-error"><AppIcon name="alert" size={18}/><span>{error}</span></div> : null}
        {saved ? <div className="form-success"><AppIcon name="check" size={18}/><span>Screening saved securely. It is now available in History.</span></div> : null}
        {syncMessage ? <div className="form-sync"><AppIcon name="wifi" size={18}/><span>{syncMessage}</span></div> : null}
        {pendingCount ? <div className="queue-controls"><span>{pendingCount} screening{pendingCount === 1 ? '' : 's'} stored on this device for up to seven days.</span><button type="button" onClick={() => { if (window.confirm('Permanently clear all pending screenings from this browser?')) void clearPendingScreenings(viewer.id ?? '').then((cleared) => { setPendingCount(0); setSyncMessage(`${cleared} pending screening${cleared === 1 ? '' : 's'} cleared from this device.`); }).catch(() => setSyncMessage('Pending data could not be cleared. Close other tabs and retry.')); }}>Clear pending data</button></div> : null}

        <button className="button button--primary button--wide" type="submit" disabled={!file || !patientCode.trim() || busy}>
          {busy ? <><span className="spinner"/>{stageLabel}</> : <><AppIcon name="scan"/>Run screening</>}
        </button>
        <p className="consent-copy">By continuing, you confirm that appropriate consent was obtained for this research screening.</p>
      </form>

      {result && preview ? <ResultPanel result={result} originalUrl={preview} patient={{ code: patientCode.trim(), age: age ? Number(age) : null, sex }}/> : <aside className="workflow-panel"><p className="eyebrow">What happens next</p><h2>One image. Four safeguards.</h2><ol><li><span>01</span><div><strong>Quality gate</strong><p>Flags blur, poor exposure, and low contrast before interpretation.</p></div></li><li><span>02</span><div><strong>DR grading</strong><p>Predicts severity from grade 0 to 4 using the active research model.</p></div></li><li><span>03</span><div><strong>DME status</strong><p>Shows a DME estimate only when the active model has a validated DME head.</p></div></li><li><span>04</span><div><strong>Visual evidence</strong><p>Shows the regions that influenced the model without calling them confirmed lesions.</p></div></li></ol></aside>}
    </div>
  );
}
