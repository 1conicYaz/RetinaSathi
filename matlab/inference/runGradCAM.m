function result = runGradCAM(network, image, classIndex, featureLayer, outputLayer)
%RUNGRADCAM Compute genuine Grad-CAM only when a compatible MATLAB network exists.
if nargin < 5 || isempty(network)
    result = struct("status", "unavailable", "reason", "Compatible network and layer names are required");
    return
end
try
    map = gradCAM(network, image, classIndex, "FeatureLayer", featureLayer, "OutputLayer", outputLayer);
    result = struct("status", "ready", "method", "gradcam", "heatmap", map, ...
        "clinicalInterpretation", "Classifier attention only; not a lesion mask.");
catch exception
    result = struct("status", "unavailable", "reason", exception.message);
end
end
