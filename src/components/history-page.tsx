import { Fragment, useCallback, useEffect, useState } from 'react';
import { useAuth } from '../lib/auth-context';
import { deleteScreening, listScreenings, submitScreeningReview, type ReviewDecision, type ScreeningRecord } from '../lib/screenings';
import { AppIcon } from './app-icon';

const gradeNames = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];

export function HistoryPage() {
  const { viewer } = useAuth();
  const [records, setRecords] = useState<ScreeningRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState<string | null>(null);
  const [reviewGrade, setReviewGrade] = useState('');
  const [reviewDecision, setReviewDecision] = useState<ReviewDecision>('reviewed');
  const [reviewNotes, setReviewNotes] = useState('');
  const [savingReview, setSavingReview] = useState(false);

  const refresh = useCallback(async () => {
    setError('');
    try { setRecords(await listScreenings()); }
    catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not load screening history.'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  async function remove(record: ScreeningRecord) {
    if (!window.confirm(`Delete screening ${record.patient_code} and its stored image?`)) return;
    setDeleting(record.id);
    setError('');
    try {
      await deleteScreening(record);
      setRecords((current) => current.filter((item) => item.id !== record.id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not delete screening.');
    } finally { setDeleting(null); }
  }

  function beginReview(record: ScreeningRecord) {
    if (reviewing === record.id) { setReviewing(null); return; }
    setReviewing(record.id);
    setReviewGrade(record.dr_grade == null ? '' : String(record.dr_grade));
    setReviewDecision('reviewed');
    setReviewNotes('');
  }

  async function saveReview(record: ScreeningRecord) {
    if (!viewer.id) return;
    if (reviewDecision === 'overridden' && reviewGrade === '') {
      setError('Choose a corrected DR grade when overriding the model.');
      return;
    }
    setSavingReview(true);
    setError('');
    try {
      await submitScreeningReview(viewer.id, record.id, reviewDecision, reviewGrade === '' ? null : Number(reviewGrade), reviewNotes);
      setReviewing(null);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not save the human review.');
    } finally { setSavingReview(false); }
  }

  return (
    <div className="history-page">
      <div className="page-heading"><div><p className="eyebrow">Screening register</p><h1>Patient history</h1><p>Only records created by your signed-in account are visible. Human reviews are append-only.</p></div><button className="button button--secondary" type="button" onClick={() => void refresh()}>Refresh</button></div>
      {error ? <div className="form-error"><AppIcon name="alert" size={18}/><span>{error}</span></div> : null}
      {loading ? <div className="empty-state"><span className="spinner spinner--dark"/><p>Loading protected records…</p></div> : records.length === 0 ? <div className="empty-state"><span className="empty-state__icon"><AppIcon name="history" size={28}/></span><h2>No screenings yet</h2><p>Your completed screenings will appear here.</p></div> : (
        <div className="history-table-wrap"><table className="history-table"><thead><tr><th>Patient</th><th>Date</th><th>DR result</th><th>DME</th><th>Quality</th><th>Referral</th><th>Review</th><th><span className="visually-hidden">Actions</span></th></tr></thead><tbody>{records.map((record) => (
          <Fragment key={record.id}>
          <tr>
            <td><strong>{record.patient_code}</strong><small>{record.patient_age ? `${record.patient_age} years` : 'Age not recorded'}</small></td>
            <td>{new Intl.DateTimeFormat('en-IN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(record.created_at))}</td>
            <td><span className={`grade-dot grade-dot--${record.dr_grade ?? 0}`}/>{record.dr_grade == null ? 'Pending' : `Grade ${record.dr_grade} · ${gradeNames[record.dr_grade]}`}</td>
            <td>{record.dme_risk == null ? '—' : `${record.dme_risk} / 2`}</td>
            <td><span className={`quality-tag quality-${record.quality_label ?? 'poor'}`}>{record.quality_label ?? 'unknown'}</span></td>
            <td><span className={`referral-tag ${record.referable_dr ? 'referral-tag--yes' : ''}`}>{record.referable_dr ? 'Refer' : 'Routine'}</span></td>
            <td><button className={`review-tag review-tag--${record.review_status}`} type="button" onClick={() => beginReview(record)}>{record.review_status.replaceAll('_', ' ')}</button></td>
            <td><button className="icon-button" type="button" disabled={deleting === record.id} onClick={() => void remove(record)} aria-label={`Delete screening ${record.patient_code}`}><AppIcon name="trash" size={17}/></button></td>
          </tr>
          {reviewing === record.id ? <tr><td className="review-cell" colSpan={8}><form className="review-form" onSubmit={(event) => { event.preventDefault(); void saveReview(record); }}><div><strong>Record a human review</strong><small>Stored as a new audit event; the model output is preserved.</small></div><label>Decision<select value={reviewDecision} onChange={(event) => setReviewDecision(event.target.value as ReviewDecision)}><option value="reviewed">Confirm reviewed</option><option value="overridden">Override grade</option><option value="inconclusive">Inconclusive</option></select></label><label>Reviewed grade<select value={reviewGrade} onChange={(event) => setReviewGrade(event.target.value)}><option value="">Not specified</option>{gradeNames.map((name, grade) => <option value={grade} key={name}>{grade} · {name}</option>)}</select></label><label className="review-notes">Notes<textarea maxLength={2000} value={reviewNotes} onChange={(event) => setReviewNotes(event.target.value)} placeholder="Clinical observations or reason for override"/></label><button className="button button--primary button--small" disabled={savingReview} type="submit">{savingReview ? 'Saving…' : 'Save review'}</button></form></td></tr> : null}
          </Fragment>
        ))}</tbody></table></div>
      )}
    </div>
  );
}
