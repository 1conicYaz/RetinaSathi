function result = runRetinaPipeline(imagePath, config)
%RUNRETINAPIPELINE Execute preprocessing, quality, inference and honest module routing.
arguments
    imagePath (1,1) string
    config (1,1) struct = struct()
end
if ~isfield(config, "modelPath"), config.modelPath = "../compute/artifacts/idrid_multitask.onnx"; end
if ~isfield(config, "lesionModelPath"), config.lesionModelPath = ""; end
config.modelPath = string(config.modelPath);
config.lesionModelPath = string(config.lesionModelPath);
if ~isfile(imagePath), error("RetinaSathi:INVALID_FILE", "Image file not found: %s", imagePath); end
try, original = imread(imagePath); catch exception
    error("RetinaSathi:CORRUPT_IMAGE", "Cannot read image: %s", exception.message);
end
[cropped, bounds] = cropFundus(original);
quality = assessImageQuality(cropped);
enhanced = enhanceFundus(cropped);
if quality.label == "poor"
    classifier = struct("status", "ungradeable", "reason", "Image quality gate requires recapture", ...
        "grade", [], "confidenceRaw", [], "confidenceCalibrated", [], "referable", []);
    lesions = struct("status", "unavailable_quality_gate", "experimental", true, "masks", []);
else
    classifier = classifyDR(enhanced, config.modelPath);
    lesions = runLesionAnalysis(enhanced, config.lesionModelPath);
end
result = struct("apiVersion", "2.0", "original", original, "cropped", cropped, ...
    "enhanced", enhanced, "cropBounds", bounds, "quality", quality, "dr", classifier, ...
    "gradcam", struct("status", "unavailable", "reason", "Call runGradCAM with validated imported network layers"), ...
    "lesions", lesions, "vessels", struct("status", "not_trained"), ...
    "opticDisc", struct("status", "not_trained"), "fovea", struct("status", "not_trained"), ...
    "requiresHumanReview", true, "disclaimer", "Screening support only; not a diagnosis.");
end
