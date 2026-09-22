#!/usr/bin/env python3
"""Build a small, blinded clinician-review queue from development predictions."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

from PIL import Image


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prediction(row: dict[str, str]) -> int:
    return max(range(5), key=lambda grade: float(row[f"p{grade}"]))


def uncertainty(row: dict[str, str]) -> float:
    probabilities = [max(float(row[f"p{grade}"]), 1e-12) for grade in range(5)]
    return -sum(value * math.log(value) for value in probabilities) / math.log(5)


def queue_candidates(
    v31_rows: list[dict[str, str]],
    v32_rows: list[dict[str, str]],
    v31_threshold: float,
    v32_threshold: float,
    total: int,
) -> list[dict[str, object]]:
    v32_by_uid = {row["image_uid"]: row for row in v32_rows}
    records: list[dict[str, object]] = []
    for left in v31_rows:
        right = v32_by_uid.get(left["image_uid"])
        if right is None:
            continue
        true_grade = int(left["true_grade"])
        pred31 = prediction(left)
        pred32 = prediction(right)
        score31 = float(left["referable_score"])
        score32 = float(right["referable_score"])
        uncertainty_score = (uncertainty(left) + uncertainty(right)) / 2
        records.append(
            {
                "image_uid": left["image_uid"],
                "source": left["source"],
                "true_grade": true_grade,
                "v31_grade": pred31,
                "v32_grade": pred32,
                "v31_referable_score": score31,
                "v32_referable_score": score32,
                "uncertainty": uncertainty_score,
                "grade_disagreement": abs(pred31 - pred32),
                "both_miss_referable": true_grade >= 2 and score31 < v31_threshold and score32 < v32_threshold,
                "v31_false_referral": true_grade < 2 and score31 >= v31_threshold,
                "v32_missed_referral": true_grade >= 2 and score32 < v32_threshold,
                "rare_grade_error": true_grade in {1, 3, 4} and (pred31 != true_grade or pred32 != true_grade),
            }
        )

    selected: list[dict[str, object]] = []
    selected_uids: set[str] = set()
    quota = max(1, total // 5)
    categories = [
        ("both models may miss referral", "both_miss_referable", lambda row: row["uncertainty"]),
        ("V3.1 may over-refer", "v31_false_referral", lambda row: row["v31_referable_score"]),
        ("V3.2 may miss referral", "v32_missed_referral", lambda row: row["uncertainty"]),
        ("models disagree on grade", "grade_disagreement", lambda row: row["grade_disagreement"] + row["uncertainty"]),
        ("rare-grade error", "rare_grade_error", lambda row: row["uncertainty"]),
    ]
    for label, flag, rank in categories:
        candidates = [row for row in records if bool(row[flag]) and row["image_uid"] not in selected_uids]
        candidates.sort(key=rank, reverse=True)
        source_grade_counts: dict[tuple[object, object], int] = {}
        taken = 0
        for row in candidates:
            key = (row["source"], row["true_grade"])
            if source_grade_counts.get(key, 0) >= 3:
                continue
            chosen = dict(row)
            chosen["selection_reason"] = label
            selected.append(chosen)
            selected_uids.add(str(row["image_uid"]))
            source_grade_counts[key] = source_grade_counts.get(key, 0) + 1
            taken += 1
            if taken >= quota or len(selected) >= total:
                break

    if len(selected) < total:
        remaining = [row for row in records if row["image_uid"] not in selected_uids]
        remaining.sort(key=lambda row: row["uncertainty"] + 0.1 * row["grade_disagreement"], reverse=True)
        for row in remaining[: total - len(selected)]:
            chosen = dict(row)
            chosen["selection_reason"] = "high uncertainty"
            selected.append(chosen)

    selected.sort(
        key=lambda row: (
            0 if row["selection_reason"] == "both models may miss referral" else 1,
            -float(row["uncertainty"]),
            str(row["image_uid"]),
        )
    )
    for index, row in enumerate(selected, start=1):
        row["review_order"] = index
        row["priority"] = "core" if index <= min(20, total) else "extended"
    return selected


def build_html(rows: list[dict[str, object]], output: Path) -> None:
    cards = []
    for row in rows:
        index = int(row["review_order"])
        uid = html.escape(str(row["image_uid"]))
        cards.append(
            f"""
            <article class="card" data-index="{index}" data-uid="{uid}">
              <div class="heading"><strong>Case {index:02d}</strong><span>{html.escape(str(row['priority']))}</span></div>
              <img src="thumbnails/{html.escape(str(row['thumbnail']))}" alt="De-identified retinal image {index}">
              <div class="fields">
                <label>Image quality
                  <select data-field="quality"><option value=""></option><option>gradeable</option><option>borderline</option><option>ungradeable</option></select>
                </label>
                <label>Doctor grade
                  <select data-field="doctor_grade"><option value=""></option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>uncertain</option></select>
                </label>
                <label>Refer?
                  <select data-field="doctor_refer"><option value=""></option><option>yes</option><option>no</option><option>uncertain</option></select>
                </label>
                <label>Visible findings<input data-field="findings" placeholder="e.g. microaneurysms, haemorrhages"></label>
                <label>Other disease suspected?<input data-field="other_disease" placeholder="optional"></label>
                <label>Short comment<textarea data-field="comment" rows="2"></textarea></label>
              </div>
              <details><summary>Reveal model and dataset information after grading</summary>
                <p>Reason selected: {html.escape(str(row['selection_reason']))}</p>
                <p>Dataset label: Grade {row['true_grade']} · V3.1: Grade {row['v31_grade']} ({float(row['v31_referable_score']):.1%} referable) · V3.2: Grade {row['v32_grade']} ({float(row['v32_referable_score']):.1%} referable)</p>
                <p>Source: {html.escape(str(row['source']))} · ID: {uid}</p>
              </details>
            </article>"""
        )
    payload = json.dumps(rows).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RetinaSathi clinician review queue</title>
<style>
body{{font-family:system-ui,sans-serif;background:#071c16;color:#e8f5ef;margin:0}}main{{max-width:1100px;margin:auto;padding:24px}}header{{position:sticky;top:0;background:#071c16ee;padding:12px 0;z-index:2}}button{{padding:10px 16px;border:0;border-radius:8px;background:#61d4b0;color:#062119;font-weight:700}}.note{{color:#b6c9c1}}.card{{background:#102f26;border:1px solid #2a5c4d;border-radius:16px;padding:16px;margin:18px 0}}.heading{{display:flex;justify-content:space-between;text-transform:capitalize}}img{{display:block;max-width:100%;height:auto;max-height:520px;margin:12px auto;border-radius:12px;background:#000}}.fields{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}label{{display:flex;flex-direction:column;gap:5px}}select,input,textarea{{font:inherit;padding:8px;border-radius:7px;border:1px solid #517b6e;background:#09231b;color:#fff}}details{{margin-top:14px;color:#cfe4dc}}.saved{{outline:2px solid #61d4b0}}
</style></head><body><main><header><h1>RetinaSathi focused clinician review</h1>
<p class="note">Review the image first. Reveal model information only after recording your judgment. Complete the 20 core cases first; the remaining cases are optional if time allows. Entries stay in this browser until exported.</p>
<button id="export">Export completed reviews as CSV</button> <span id="status"></span></header>
{''.join(cards)}
<script>
const rows={payload}; const key='retinasathi-clinician-review-v1'; let answers=JSON.parse(localStorage.getItem(key)||'{{}}');
function save(){{document.querySelectorAll('.card').forEach(card=>{{const uid=card.dataset.uid;answers[uid]=answers[uid]||{{}};card.querySelectorAll('[data-field]').forEach(x=>answers[uid][x.dataset.field]=x.value);card.classList.toggle('saved',Boolean(answers[uid].doctor_grade||answers[uid].quality));}});localStorage.setItem(key,JSON.stringify(answers));document.getElementById('status').textContent=Object.values(answers).filter(x=>x.doctor_grade||x.quality).length+' cases started';}}
document.querySelectorAll('.card').forEach(card=>{{const a=answers[card.dataset.uid]||{{}};card.querySelectorAll('[data-field]').forEach(x=>{{x.value=a[x.dataset.field]||'';x.addEventListener('change',save);x.addEventListener('input',save);}});}});save();
document.getElementById('export').onclick=()=>{{save();const fields=['review_order','priority','image_uid','quality','doctor_grade','doctor_refer','findings','other_disease','comment'];const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';const lines=[fields.join(',')];rows.forEach(r=>{{const a=answers[r.image_uid]||{{}};lines.push(fields.map(f=>esc(f in r?r[f]:a[f])).join(','));}});const blob=new Blob([lines.join('\n')],{{type:'text/csv'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='retinasathi_clinician_reviews.csv';link.click();URL.revokeObjectURL(link.href);}};
</script></main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/v3_data/v3_initial_manifest.csv"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v31-run", type=Path, required=True)
    parser.add_argument("--v32-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing review queue: {args.output}")
    manifest = {row["image_uid"]: row for row in read_csv(args.manifest)}
    final31 = json.loads((args.v31_run / "final_validation.json").read_text())
    final32 = json.loads((args.v32_run / "final_validation.json").read_text())
    rows = queue_candidates(
        read_csv(args.v31_run / "validation_predictions.csv"),
        read_csv(args.v32_run / "validation_predictions.csv"),
        float(final31["threshold_gate"]["selected"]["threshold"]),
        float(final32["threshold_gate"]["selected"]["threshold"]),
        args.total,
    )
    args.output.mkdir(parents=True)
    thumbnails = args.output / "thumbnails"
    thumbnails.mkdir()
    for row in rows:
        record = manifest[str(row["image_uid"])]
        source = args.data_root / record["relative_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        filename = f"case_{int(row['review_order']):02d}.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            image.save(thumbnails / filename, quality=90, optimize=True)
        row["thumbnail"] = filename
        row["relative_path"] = record["relative_path"]
    fields = list(rows[0]) if rows else []
    with (args.output / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.output / "README.md").write_text(
        "# Focused clinician review queue\n\n"
        "Open `index.html` in a browser. Review the image before revealing model information. "
        "Complete the 20 core cases first. Browser entries are stored locally; use the export button when finished. "
        "These derived thumbnails remain subject to their source dataset terms and must not be redistributed.\n",
        encoding="utf-8",
    )
    build_html(rows, args.output)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "core_cases": min(20, len(rows))}, indent=2))


if __name__ == "__main__":
    main()
