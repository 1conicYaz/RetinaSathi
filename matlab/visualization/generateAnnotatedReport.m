function figureHandle = generateAnnotatedReport(result)
%GENERATEANNOTATEDREPORT Render judge-facing, medically cautious pipeline output.
figureHandle = figure("Name", "RetinaSathi AI Screening Support Report", "Color", "white");
tiledlayout(2, 2);
nexttile; imshow(result.original); title("Original fundus", "Color", "black");
if isfield(result, "enhanced")
    enhanced = result.enhanced;
elseif isfield(result, "preprocessing") && isfield(result.preprocessing, "enhanced")
    enhanced = result.preprocessing.enhanced;
else
    enhanced = result.original;
end
nexttile; imshow(enhanced); title("Enhanced fundus", "Color", "black");
nexttile([1 2]); axis off
gradeText = "unavailable"; referableText = "unavailable"; confidenceText = "unavailable";
if isfield(result.dr, "grade") && ~isempty(result.dr.grade), gradeText = string(result.dr.grade); end
if isfield(result.dr, "referable") && ~isempty(result.dr.referable), referableText = string(logical(result.dr.referable)); end
if isfield(result.dr, "confidenceRaw") && ~isempty(result.dr.confidenceRaw), confidenceText = sprintf("%.1f%% raw", 100*result.dr.confidenceRaw); end
if isfield(result.dr, "confidence") && ~isempty(result.dr.confidence), confidenceText = sprintf("%.1f%% calibrated", 100*result.dr.confidence); end
summary = sprintf("AI SCREENING SUPPORT REPORT — NOT A DIAGNOSIS\nQuality: %s (%.2f)\nDR grade: %s · module: %s\nReferable: %s · confidence: %s\nHuman review: required\n%s", ...
    result.quality.label, result.quality.score, gradeText, result.dr.status, referableText, confidenceText, result.disclaimer);
text(0.02, 0.9, summary, "FontSize", 13, "VerticalAlignment", "top", "Color", "black");
end
