function window = launchV34LesionPrototype(initialImagePath)
%LAUNCHV34LESIONPROTOTYPE Open the V3.4 app with V3.2 lesion evidence enabled.
% The lesion overlay is an experimental prototype and cannot confirm lesions.
arguments
    initialImagePath (1,1) string = ""
end
root=fileparts(mfilename("fullpath"));
setup;
cfg=struct("modelGeneration","v3.4", ...
    "modelPath",fullfile(root,"..","models","retinasathi-v3-4.onnx"), ...
    "manifestPath",fullfile(root,"..","models","retinasathi-v3-4.manifest.json"), ...
    "lesionModelPath",fullfile(root,"..","models","retinasathi-lesions-v3-2.onnx"), ...
    "lesionManifestPath",fullfile(root,"..","models","retinasathi-lesions-v3-2.manifest.json"));
if strlength(initialImagePath)>0, cfg.initialImagePath=initialImagePath; end
required=[string(cfg.modelPath),string(cfg.manifestPath),string(cfg.lesionModelPath),string(cfg.lesionManifestPath)];
if any(~isfile(required))
    error("RetinaSathi:MissingPrototypeArtifact","One or more V3.4/V3.2 model artifacts are missing.");
end
window=launchRetinaSathiApp(cfg);
end
