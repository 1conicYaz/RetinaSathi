#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

matlab_bin="${MATLAB_BIN:-/Applications/MATLAB_R2026a.app/bin/matlab}"
model="${V3_4_ONNX:-models/retinasathi-v3-4.onnx}"
manifest="${V3_4_MANIFEST:-models/retinasathi-v3-4.manifest.json}"
fixtures="${V3_4_PARITY_FIXTURES:-runs/v3_4_matlab_parity}"

[[ -x "$matlab_bin" ]] || { echo "MATLAB executable not found: $matlab_bin" >&2; exit 2; }
[[ -f "$model" ]] || { echo "V3.4 ONNX model not found: $model" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "V3.4 manifest not found: $manifest" >&2; exit 2; }
[[ -f "$fixtures/fixture_manifest.json" ]] || { echo "Parity fixtures not found: $fixtures" >&2; exit 2; }

"$matlab_bin" -batch "cd('matlab'); setup; results=runtests('tests'); assert(all([results.Passed])); report=runV34Parity('../$fixtures','../$model','../$manifest','../$fixtures/matlab_parity_report.json'); assert(report.passed); fprintf('PASS: %d MATLAB tests, %d parity cases, grade %.3f, referral %.3f\\n',numel(results),numel(report.cases),report.gradeAgreementRate,report.referralAgreementRate);"
