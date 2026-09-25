function tests = testV34Contract
tests = functiontests(localfunctions);
end

function testOutputContract(testCase)
image = zeros(120, 180, 3, "uint8");
image(15:105, 25:155, :) = 80;
[tensor, audit] = preprocessV34Fundus(image);
verifySize(testCase, tensor, [392 392 3]);
verifyClass(testCase, tensor, "single");
verifyEqual(testCase, audit.contract, "retinasathi-v3.4-preprocessing-v1");
verifyEqual(testCase, audit.colorNormalization, false);
verifyTrue(testCase, all(isfinite(tensor), "all"));
end

function testBlackImageStaysFinite(testCase)
[tensor, audit] = preprocessV34Fundus(zeros(64, 64, "uint8"));
verifyTrue(testCase, all(isfinite(tensor), "all"));
verifyEqual(testCase, audit.cropBounds, [1 1 64 64]);
end

function testCalibrationPolicy(testCase)
calibration = struct("ordinalTemperature", 1, "nominalTemperature", 1, ...
    "binaryTemperature", 1, "referableThreshold", 0.5);
result = applyV34Calibration(2, [3 2 1 0], [0 0 0 2 0], calibration);
verifyEqual(testCase, numel(result.gradeProbabilities), 5);
verifyEqual(testCase, sum(result.gradeProbabilities), 1, "AbsTol", 1e-12);
verifyTrue(testCase, result.referable);
verifyEqual(testCase, result.dmeStatus, "not_assessed");
verifyTrue(testCase, result.requiresHumanReview);
end

function testPillowGaussianContract(testCase)
image = zeros(31, 31, 3, "uint8");
image(16, 16, :) = 255;
blurred = pillowGaussianBlur(image, 3.5, 3);
verifySize(testCase, blurred, size(image));
verifyClass(testCase, blurred, "uint8");
verifyGreaterThan(testCase, blurred(16,16,1), uint8(0));
verifyLessThan(testCase, blurred(16,16,1), uint8(255));
verifyEqual(testCase, blurred(1,1,1), uint8(0));
end

function testBinarySha256(testCase)
path = string(tempname);
cleanup = onCleanup(@() deleteIfPresent(path)); %#ok<NASGU>
file = fopen(path, "w");
fwrite(file, uint8('abc'), "uint8");
fclose(file);
verifyEqual(testCase, sha256File(path), ...
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
end

function testLesionModuleIsExplicitWhenModelMissing(testCase)
result = runLesionAnalysis(zeros(64,64,3,"uint8"),"missing-lesion-model.onnx","");
verifyEqual(testCase,result.status,"not_trained");
verifyTrue(testCase,result.experimental);
verifyEmpty(testCase,result.masks);
end

function deleteIfPresent(path)
if isfile(path), delete(path); end
end
