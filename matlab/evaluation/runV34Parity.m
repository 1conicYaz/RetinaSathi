function report = runV34Parity(fixtureDirectory, modelPath, manifestPath, outputPath)
%RUNV34PARITY Compare Python, ONNX and MATLAB on fixed local retinal cases.
arguments
    fixtureDirectory (1,1) string
    modelPath (1,1) string
    manifestPath (1,1) string
    outputPath (1,1) string = ""
end
inputs = dir(fullfile(fixtureDirectory, "case_*_input.*"));
if isempty(inputs), error("RetinaSathi:NoParityCases", "No parity cases found in %s", fixtureDirectory); end
records = repmat(struct(), 0, 1);
for index = 1:numel(inputs)
    token = extractBefore(string(inputs(index).name), "_input");
    referencePath = fullfile(fixtureDirectory, token + "_python_reference.mat");
    if ~isfile(referencePath), error("RetinaSathi:MissingReference", "Missing %s", referencePath); end
    reference = load(referencePath);
    image = imread(fullfile(inputs(index).folder, inputs(index).name));
    [matlabTensor, audit] = preprocessV34Fundus(image);
    tensorDifference = abs(double(matlabTensor) - double(reference.normalizedHWC));
    prediction = classifyDRV34(matlabTensor, modelPath, manifestPath);
    record = struct("caseName", token, "meanTensorAbsDifference", mean(tensorDifference, "all"), ...
        "maxTensorAbsDifference", max(tensorDifference, [], "all"), ...
        "matlabStatus", prediction.status, "inputSize", audit.inputSize, ...
        "onnxReferableLogitAbsDifference", NaN, "onnxOrdinalLogitMaxAbsDifference", NaN, ...
        "onnxNominalLogitMaxAbsDifference", NaN, "matlabGrade", NaN, ...
        "pythonTensorGradeThroughMatlabONNX", NaN, "gradeAgreement", false, ...
        "referralAgreement", false, "failureReason", "");
    if prediction.status ~= "unavailable"
        referenceCalibration = classifyDRV34(reference.normalizedHWC, modelPath, manifestPath);
        record.onnxReferableLogitAbsDifference = abs(double(prediction.rawLogits.referable(1)) - double(reference.referableLogit(1)));
        record.onnxOrdinalLogitMaxAbsDifference = max(abs(double(prediction.rawLogits.ordinal(:)) - double(reference.ordinalLogits(:))));
        record.onnxNominalLogitMaxAbsDifference = max(abs(double(prediction.rawLogits.nominal(:)) - double(reference.nominalLogits(:))));
        record.matlabGrade = prediction.grade;
        record.pythonTensorGradeThroughMatlabONNX = referenceCalibration.grade;
        record.gradeAgreement = prediction.grade == referenceCalibration.grade;
        record.referralAgreement = prediction.referable == referenceCalibration.referable;
    elseif isfield(prediction, "reason")
        record.failureReason = string(prediction.reason);
    end
    if isempty(records), records = record; else, records(end+1, 1) = record; end %#ok<AGROW>
end
available = arrayfun(@(item) string(item.matlabStatus) ~= "unavailable", records);
report = struct("schemaVersion", 1, "generatedAt", string(datetime("now", "TimeZone", "UTC")), ...
    "fixtureDirectory", fixtureDirectory, "modelPath", modelPath, "manifestPath", manifestPath, ...
    "cases", records, "allInferenceAvailable", all(available), ...
    "acceptance", struct("requiredGradeAgreement", 1.0, "requiredReferralAgreement", 1.0, ...
        "note", "Pixel differences are reported because PIL and MATLAB filters may differ; decision agreement is mandatory."));
if all(available)
    report.gradeAgreementRate = mean([records.gradeAgreement]);
    report.referralAgreementRate = mean([records.referralAgreement]);
    report.passed = report.gradeAgreementRate == 1 && report.referralAgreementRate == 1;
else
    report.gradeAgreementRate = NaN;
    report.referralAgreementRate = NaN;
    report.passed = false;
end
if strlength(outputPath) > 0
    file = fopen(outputPath, "w");
    if file < 0, error("RetinaSathi:WriteFailed", "Cannot write %s", outputPath); end
    cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
    fwrite(file, jsonencode(report, "PrettyPrint", true));
end
disp(report)
end
