function result = runLesionAnalysis(image, modelPath)
%RUNLESIONANALYSIS Never fabricate lesions when a trained model is absent.
if nargin < 2 || ~isfile(modelPath)
    result = struct("status", "not_trained", "experimental", true, "masks", []);
    return
end
loaded = load(modelPath, "network", "modelVersion");
labels = semanticseg(image, loaded.network);
result = struct("status", "ready", "experimental", true, "masks", labels, ...
    "modelVersion", loaded.modelVersion);
end
