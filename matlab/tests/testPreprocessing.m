function tests = testPreprocessing
tests = functiontests(localfunctions);
end

function testBlackFrameIsRejected(testCase)
quality = assessImageQuality(zeros(256, 256, 3, "uint8"));
verifyEqual(testCase, quality.label, "poor");
end

function testBorderIsRemoved(testCase)
image = zeros(100, 100, 3, "uint8"); image(20:80, 15:85, :) = 100;
[cropped, ~] = cropFundus(image);
verifyLessThan(testCase, size(cropped, 1), size(image, 1));
verifyLessThan(testCase, size(cropped, 2), size(image, 2));
end

function testPoorQualityStopsClinicalInference(testCase)
temporaryPath = string(tempname) + ".png";
cleanup = onCleanup(@() deleteIfPresent(temporaryPath)); %#ok<NASGU>
imwrite(zeros(256, 256, 3, "uint8"), temporaryPath);
result = runRetinaPipeline(temporaryPath);
verifyEqual(testCase, result.quality.label, "poor");
verifyEqual(testCase, result.dr.status, "ungradeable");
verifyEmpty(testCase, result.dr.grade);
verifyEqual(testCase, result.lesions.status, "unavailable_quality_gate");
end

function deleteIfPresent(path)
if isfile(path), delete(path); end
end
