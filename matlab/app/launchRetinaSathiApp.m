function window = launchRetinaSathiApp(config)
%LAUNCHRETINASATHIAPP Programmatic App Designer-compatible demonstration UI.
arguments
    config (1,1) struct = struct()
end
if ~isfield(config, "modelGeneration"), config.modelGeneration = "baseline"; end
if ~isfield(config, "visible"), config.visible = "on"; end
if string(config.modelGeneration) == "v3.4"
    if ~isfield(config, "modelPath"), config.modelPath = "../models/retinasathi-v3-4.onnx"; end
    if ~isfield(config, "manifestPath"), config.manifestPath = "../models/retinasathi-v3-4.manifest.json"; end
else
    if ~isfield(config, "modelPath"), config.modelPath = "../compute/artifacts/idrid_multitask.onnx"; end
end
if ~isfield(config, "lesionModelPath"), config.lesionModelPath = ""; end
window = uifigure("Name", "RetinaSathi — Screening Support", "Position", [100 100 1120 700], ...
    "Visible", string(config.visible));
grid = uigridlayout(window, [3 4]); grid.RowHeight = {45, "1x", 150};
titleLabel = uilabel(grid, "Text", "RetinaSathi MATLAB Screening Support · " + string(config.modelGeneration), ...
    "FontSize", 20, "FontWeight", "bold");
titleLabel.Layout.Column = [1 3];
status = uilabel(grid, "Text", "Choose a fundus image · Human review required", "HorizontalAlignment", "right");
status.Layout.Column = 4;
originalAxes = uiaxes(grid); originalAxes.Layout.Row = 2; originalAxes.Layout.Column = [1 2]; title(originalAxes, "Original fundus");
enhancedAxes = uiaxes(grid); enhancedAxes.Layout.Row = 2; enhancedAxes.Layout.Column = [3 4]; title(enhancedAxes, "Enhanced fundus");
details = uitextarea(grid, "Editable", "off", "Value", "No result yet."); details.Layout.Row = 3; details.Layout.Column = [1 3];
button = uibutton(grid, "Text", "Choose and analyze image"); button.Layout.Row = 3; button.Layout.Column = 4;
button.ButtonPushedFcn = @analyze;
if isfield(config, "initialImagePath") && strlength(string(config.initialImagePath)) > 0
    drawnow;
    analyzePath(string(config.initialImagePath));
end
    function analyze(~, ~)
        [name, folder] = uigetfile({"*.jpg;*.jpeg;*.png;*.tif;*.tiff", "Fundus images"});
        if isequal(name, 0), return; end
        analyzePath(string(fullfile(folder, name)));
    end
    function analyzePath(imagePath)
        try
            if string(config.modelGeneration) == "v3.4"
                result = runRetinaPipelineV34(imagePath, config);
                enhanced = result.preprocessing.enhanced;
            else
                result = runRetinaPipeline(imagePath, config);
                enhanced = result.enhanced;
            end
            imshow(result.original, "Parent", originalAxes); imshow(enhanced, "Parent", enhancedAxes);
            gradeText = "unavailable"; referableText = "unavailable"; confidenceText = "unavailable";
            if isfield(result.dr, "grade") && ~isempty(result.dr.grade), gradeText = string(result.dr.grade); end
            if isfield(result.dr, "referable") && ~isempty(result.dr.referable), referableText = string(logical(result.dr.referable)); end
            if isfield(result.dr, "confidenceRaw") && ~isempty(result.dr.confidenceRaw), confidenceText = sprintf("%.1f%% raw", 100*result.dr.confidenceRaw); end
            if isfield(result.dr, "confidence") && ~isempty(result.dr.confidence), confidenceText = sprintf("%.1f%% calibrated", 100*result.dr.confidence); end
            lesionStatus = moduleStatus(result, "lesions", "not_integrated_in_matlab");
            vesselStatus = moduleStatus(result, "vessels", "not_integrated_in_matlab");
            discStatus = moduleStatus(result, "opticDisc", "not_integrated_in_matlab");
            foveaStatus = moduleStatus(result, "fovea", "not_integrated_in_matlab");
            dmeStatus = moduleStatus(result, "dme", "legacy_head");
            probabilityText = "unavailable";
            if isfield(result.dr, "gradeProbabilities") && numel(result.dr.gradeProbabilities) == 5
                probabilityText = strjoin(compose("G%d %.1f%%", (0:4)', 100*result.dr.gradeProbabilities(:)), " · ");
            end
            referralText = "unavailable";
            if isfield(result.dr, "referableScore") && isfield(result.dr, "threshold")
                referralText = sprintf("%.1f%% (threshold %.1f%%)", 100*result.dr.referableScore, 100*result.dr.threshold);
            end
            recommendation = "Human review required because V3.4 is a research candidate.";
            if result.quality.label == "poor"
                recommendation = "Retake the image before interpretation.";
            elseif isfield(result.dr, "referable") && logical(result.dr.referable)
                recommendation = "Refer to an ophthalmologist for confirmatory examination.";
            elseif isfield(result.dr, "confidence") && result.dr.confidence < 0.60
                recommendation = "Low-confidence result: clinician review or recapture required.";
            end
            details.Value = [string(sprintf("Quality: %s (%.0f%%)", result.quality.label, 100*result.quality.score)); ...
                string(sprintf("DR grade: %s · module: %s", gradeText, result.dr.status)); ...
                string(sprintf("Grade confidence: %s (highest calibrated grade probability; not accuracy)", confidenceText)); ...
                "Grade probabilities: " + probabilityText; ...
                string(sprintf("Referable: %s · score: %s", referableText, referralText)); ...
                string(sprintf("DME: %s · V3.4 has no DME prediction head", dmeStatus)); ...
                string(sprintf("Lesions: %s · Vessels: %s", lesionStatus, vesselStatus)); ...
                string(sprintf("Optic disc: %s · Fovea: %s", discStatus, foveaStatus)); ...
                string(sprintf("Attention: %s · MATLAB V3.4 attention parity is not validated", result.gradcam.status)); ...
                "Recommendation: " + recommendation; ...
                "Review status: pending"; string(result.disclaimer)];
            status.Text = "Analysis complete · Check module statuses";
        catch exception
            status.Text = "Analysis failed";
            if string(window.Visible) == "on"
                uialert(window, exception.message, "RetinaSathi error");
            else
                rethrow(exception);
            end
        end
    end
end

function status = moduleStatus(result, field, fallback)
if isfield(result, field) && isstruct(result.(field)) && isfield(result.(field), "status")
    status = string(result.(field).status);
else
    status = string(fallback);
end
end
