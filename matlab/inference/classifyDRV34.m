function result = classifyDRV34(normalizedHWC, modelPath, manifestPath)
%CLASSIFYDRV34 Import V3.4 ONNX and apply its hash-bound calibration policy.
if ~isfile(modelPath), result = struct("status", "unavailable", "reason", "V3.4 ONNX file not found"); return; end
if ~isfile(manifestPath), result = struct("status", "unavailable", "reason", "V3.4 manifest file not found"); return; end
try
    manifest = jsondecode(fileread(manifestPath));
    calibration = struct( ...
        "ordinalTemperature", manifest.calibration.ordinal_temperature, ...
        "nominalTemperature", manifest.calibration.nominal_temperature, ...
        "binaryTemperature", manifest.calibration.binary_temperature, ...
        "referableThreshold", manifest.calibration.referable_threshold);
    expectedDigest = lower(string(manifest.onnx.sha256));
    persistent net loadedPath
    if isempty(net) || loadedPath ~= string(modelPath)
        actualDigest = sha256File(string(modelPath));
        if actualDigest ~= expectedDigest
            error("RetinaSathi:ArtifactIntegrity", ...
                "V3.4 ONNX SHA-256 mismatch: expected %s, got %s", expectedDigest, actualDigest);
        end
        net = importNetworkFromONNX(modelPath, "InputDataFormats", "BCSS", ...
            "OutputDataFormats", ["BC" "BC" "BC"]);
        loadedPath = string(modelPath);
    end
    outputs = cell(1, 3);
    [outputs{:}] = predict(net, reshape(single(normalizedHWC), size(normalizedHWC,1), size(normalizedHWC,2), 3, 1));
    referableLogit = extractdata(outputs{1});
    ordinalLogits = extractdata(outputs{2});
    nominalLogits = extractdata(outputs{3});
    result = applyV34Calibration(referableLogit, ordinalLogits, nominalLogits, calibration);
    result.rawLogits = struct("referable", referableLogit, "ordinal", ordinalLogits, "nominal", nominalLogits);
    result.modelVersion = manifest.model_version;
    result.onnxSha256 = expectedDigest;
catch exception
    result = struct("status", "unavailable", "reason", exception.message);
end
end
