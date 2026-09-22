function metrics = evaluateDR(truth, predicted)
%EVALUATEDR Accuracy, macro F1, confusion matrix, and quadratic kappa.
truth = double(truth(:)); predicted = double(predicted(:));
classes = 0:4; confusion = confusionmat(truth, predicted, "Order", classes);
precision = diag(confusion) ./ max(sum(confusion, 1)', 1);
recall = diag(confusion) ./ max(sum(confusion, 2), 1);
f1 = 2 .* precision .* recall ./ max(precision + recall, eps);
n = sum(confusion, "all"); observed = confusion / max(n, 1);
expected = (sum(confusion, 2) * sum(confusion, 1)) / max(n^2, 1);
[row, column] = ndgrid(classes, classes); weights = ((row-column)/4).^2;
qwk = 1 - sum(weights.*observed, "all") / max(sum(weights.*expected, "all"), eps);
metrics = struct("accuracy", mean(truth == predicted), "macroF1", mean(f1), ...
    "qwk", qwk, "confusionMatrix", confusion);
end
