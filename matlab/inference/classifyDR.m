function result = classifyDR(image, modelPath)
%CLASSIFYDR Import and run the preserved ONNX baseline with explicit limitations.
if ~isfile(modelPath)
    result = struct("status", "unavailable", "reason", "ONNX model file not found");
    return
end
try
    persistent net loadedPath
    if isempty(net) || loadedPath ~= string(modelPath)
        net = importNetworkFromONNX(modelPath, "InputDataFormats", "BCSS");
        loadedPath = string(modelPath);
    end
    input = im2single(imresize(image, [224 224]));
    input = (input - reshape(single([0.485 0.456 0.406]), 1, 1, 3)) ./ ...
        reshape(single([0.229 0.224 0.225]), 1, 1, 3);
    outputs = predict(net, input);
    if iscell(outputs), gradeLogits = outputs{1}; else, gradeLogits = outputs(:, 1:5); end
    % importNetworkFromONNX returns numeric arrays here, so DataFormat is
    % invalid. Softmax uses the first non-singleton (grade) dimension for
    % either the observed 1x5 or a compatible 5x1 output.
    probabilities = softmax(gradeLogits);
    [confidence, index] = max(extractdata(probabilities(:)));
    result = struct("status", "baseline_v1", "grade", index-1, ...
        "confidenceRaw", confidence, "confidenceCalibrated", NaN, ...
        "referable", index-1 >= 2, "modelVersion", "idrid-mobilenetv3-multitask-v0.1");
catch exception
    result = struct("status", "unavailable", "reason", exception.message);
end
end
