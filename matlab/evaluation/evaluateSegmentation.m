function metrics = evaluateSegmentation(truthMask, predictedMask)
%EVALUATESEGMENTATION Binary Dice and IoU with empty-mask-safe denominators.
truth = logical(truthMask); predicted = logical(predictedMask);
intersection = nnz(truth & predicted); union = nnz(truth | predicted);
metrics = struct("dice", 2*intersection/max(nnz(truth)+nnz(predicted), 1), ...
    "iou", intersection/max(union, 1));
end
