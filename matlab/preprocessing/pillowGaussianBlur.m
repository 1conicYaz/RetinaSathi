function output = pillowGaussianBlur(image, sigma, passes)
%PILLOWGAUSSIANBLUR Match Pillow 12 GaussianBlur's extended box filters.
% Pillow converts the requested Gaussian sigma into one fractional box radius
% and applies that box three times horizontally and vertically, rounding to
% uint8 after every pass. Keeping this behavior avoids deployment shift.
arguments
    image {mustBeNumeric}
    sigma (1,1) double {mustBeNonnegative}
    passes (1,1) double {mustBeInteger,mustBePositive} = 3
end
if sigma == 0, output = im2uint8(image); return; end
sigmaSquaredPerPass = sigma^2 / passes;
boxLength = sqrt(12 * sigmaSquaredPerPass + 1);
integerRadius = floor((boxLength - 1) / 2);
fraction = (2 * integerRadius + 1) * ...
    (integerRadius * (integerRadius + 1) - 3 * sigmaSquaredPerPass);
fraction = fraction / (6 * (sigmaSquaredPerPass - (integerRadius + 1)^2));
floatRadius = integerRadius + fraction;
radius = floor(floatRadius);

scale = 2^24;
wholeWeight = floor(scale / (2 * floatRadius + 1));
farWeight = floor((scale - (2 * radius + 1) * wholeWeight) / 2);
kernel = double([farWeight repmat(wholeWeight, 1, 2 * radius + 1) farWeight]) / scale;

output = im2uint8(image);
for pass = 1:passes
    output = uint8(floor(imfilter(double(output), kernel, "replicate", "same", "corr") + 0.5));
end
verticalKernel = kernel(:);
for pass = 1:passes
    output = uint8(floor(imfilter(double(output), verticalKernel, "replicate", "same", "corr") + 0.5));
end
end
