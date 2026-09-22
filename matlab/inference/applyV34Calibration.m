function result = applyV34Calibration(referableLogit, ordinalLogits, nominalLogits, calibration)
%APPLYV34CALIBRATION Apply the frozen V3.4 temperatures, fusion and threshold.
required = ["ordinalTemperature", "nominalTemperature", "binaryTemperature", "referableThreshold"];
for key = required
    if ~isfield(calibration, key), error("RetinaSathi:MissingCalibration", "Missing %s", key); end
end
ordinal = double(ordinalLogits(:)') / calibration.ordinalTemperature;
cumulative = 1 ./ (1 + exp(-ordinal));
for index = 2:numel(cumulative)
    cumulative(index) = min(cumulative(index), cumulative(index-1));
end
endpoints = [1 cumulative 0];
ordinalProbabilities = max(0, endpoints(1:end-1) - endpoints(2:end));
ordinalProbabilities = ordinalProbabilities / max(sum(ordinalProbabilities), eps);
nominal = double(nominalLogits(:)') / calibration.nominalTemperature;
nominal = exp(nominal - max(nominal));
nominalProbabilities = nominal / sum(nominal);
gradeProbabilities = 0.5 * ordinalProbabilities + 0.5 * nominalProbabilities;
[confidence, gradeIndex] = max(gradeProbabilities);
referableScore = 1 / (1 + exp(-double(referableLogit(1)) / calibration.binaryTemperature));
result = struct("grade", gradeIndex - 1, "gradeProbabilities", gradeProbabilities, ...
    "confidence", confidence, "referableScore", referableScore, ...
    "referable", referableScore >= calibration.referableThreshold, ...
    "threshold", calibration.referableThreshold, "dmeStatus", "not_assessed", ...
    "requiresHumanReview", true, "status", "candidate", ...
    "disclaimer", "Research screening support only; not a diagnosis.");
end
