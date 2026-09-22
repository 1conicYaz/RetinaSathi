function quality = assessImageQuality(image)
%ASSESSIMAGEQUALITY Deterministic hard safety checks; not a learned quality model.
gray = im2double(im2gray(imresize(image, [256 256])));
[xx, yy] = meshgrid(1:256, 1:256);
mask = gray > 0.045 & (xx - 128.5).^2 + (yy - 128.5).^2 < 121^2;
pixels = gray(mask);
if numel(pixels) < 100
    quality = struct("status", "heuristic", "label", "poor", "score", 0, ...
        "issues", {{"Retina is not visible"}}, "recaptureGuidance", "Center the fundus and retake the image.");
    return
end
brightness = mean(pixels); contrast = std(pixels);
[gx, gy] = gradient(gray); sharpness = mean(hypot(gx(mask), gy(mask)));
exposure = max(0, 1 - abs(brightness - 0.38) / 0.32);
score = 0.35*exposure + 0.25*min(1, contrast/0.08) + 0.40*min(1, sharpness/0.008);
issues = {};
if brightness < 0.22, issues{end+1} = "Image is too dark"; end %#ok<AGROW>
if brightness > 0.58, issues{end+1} = "Image is overexposed"; end %#ok<AGROW>
if contrast < 0.045, issues{end+1} = "Retinal contrast is low"; end %#ok<AGROW>
if sharpness < 0.0045, issues{end+1} = "Image may be blurred"; end %#ok<AGROW>
if score >= 0.68, label = "good"; elseif score >= 0.45, label = "usable"; else, label = "poor"; end
guidance = "No recapture required by deterministic checks.";
if label == "poor", guidance = "Retake after correcting exposure, focus, and fundus centering."; end
quality = struct("status", "heuristic", "label", label, "score", score, ...
    "brightness", brightness, "contrast", contrast, "sharpness", sharpness, ...
    "issues", {issues}, "recaptureGuidance", guidance);
end
