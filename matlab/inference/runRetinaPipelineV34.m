function result = runRetinaPipelineV34(imagePath, config)
%RUNRETINAPIPELINEV34 Execute the native MATLAB V3.4 research-candidate path.
arguments
    imagePath (1,1) string
    config (1,1) struct = struct()
end
if ~isfield(config, "modelPath"), config.modelPath = "../models/retinasathi-v3-4.onnx"; end
if ~isfield(config, "manifestPath"), config.manifestPath = "../models/retinasathi-v3-4.manifest.json"; end
if ~isfile(imagePath), error("RetinaSathi:INVALID_FILE", "Image file not found: %s", imagePath); end
try, original = imread(imagePath); catch exception
    error("RetinaSathi:CORRUPT_IMAGE", "Cannot read image: %s", exception.message);
end
[croppedForQuality, ~] = cropFundus(original);
quality = assessImageQuality(croppedForQuality);
[normalizedHWC, preprocessing] = preprocessV34Fundus(original);
if quality.label == "poor"
    classifier = struct("status", "ungradeable", "reason", "Image quality gate requires recapture", ...
        "grade", [], "confidence", [], "referable", [], "dmeStatus", "not_assessed");
else
    classifier = classifyDRV34(normalizedHWC, string(config.modelPath), string(config.manifestPath));
end
result = struct("apiVersion", "2.0", "modelGeneration", "v3.4", "original", original, ...
    "quality", quality, "preprocessing", preprocessing, "dr", classifier, ...
    "gradcam", struct("status", "unavailable", "reason", "MATLAB V3.4 attention parity is not yet validated"), ...
    "dme", struct("status", "not_assessed", "reason", "V3.4 has no DME head"), ...
    "requiresHumanReview", true, "disclaimer", "Research screening support only; not a diagnosis.");
end
