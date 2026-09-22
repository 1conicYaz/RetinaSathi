function [normalizedHWC, audit] = preprocessV34Fundus(image, options)
%PREPROCESSV34FUNDUS Reproduce the frozen V3.4 retinal input contract.
arguments
    image {mustBeNumeric}
    options.InputSize (1,1) double {mustBeInteger,mustBePositive} = 392
end

if ndims(image) == 2
    image = repmat(image, 1, 1, 3);
elseif size(image, 3) > 3
    image = image(:, :, 1:3);
end
rgb = im2uint8(image);
[height, width, ~] = size(rgb);
visible = max(rgb, [], 3) > 8;
rows = find(any(visible, 2));
columns = find(any(visible, 1));
if isempty(rows) || isempty(columns)
    top = 1; bottom = height; left = 1; right = width;
else
    padding = round(min(height, width) * 0.01);
    top = max(1, rows(1) - padding);
    bottom = min(height, rows(end) + padding);
    left = max(1, columns(1) - padding);
    right = min(width, columns(end) + padding);
end
cropped = rgb(top:bottom, left:right, :);

side = max(size(cropped, 1), size(cropped, 2));
square = zeros(side, side, 3, "uint8");
rowOffset = floor((side - size(cropped, 1)) / 2);
columnOffset = floor((side - size(cropped, 2)) / 2);
square(rowOffset + (1:size(cropped, 1)), columnOffset + (1:size(cropped, 2)), :) = cropped;

resized = imresize(square, [options.InputSize options.InputSize], "lanczos3", "Antialiasing", true);
sigma = max(1, options.InputSize * 0.035);
blurred = pillowGaussianBlur(resized, sigma, 3);
enhanced = uint8(floor(min(max(4 * single(resized) - 4 * single(blurred) + 128, 0), 255)));

meanRGB = reshape(single([0.485 0.456 0.406]), 1, 1, 3);
stdRGB = reshape(single([0.229 0.224 0.225]), 1, 1, 3);
normalizedHWC = (single(enhanced) / 255 - meanRGB) ./ stdRGB;
audit = struct( ...
    "contract", "retinasathi-v3.4-preprocessing-v1", ...
    "cropBounds", [left top right-left+1 bottom-top+1], ...
    "cropped", cropped, "squarePadded", square, "resized", resized, ...
    "enhanced", enhanced, "gaussianSigma", sigma, "gaussianImplementation", "Pillow extended box, 3 passes", ...
    "inputSize", options.InputSize, "normalizationMeanRGB", meanRGB, ...
    "normalizationStdRGB", stdRGB, "colorNormalization", false, ...
    "modelLayout", "SSCB");
end
