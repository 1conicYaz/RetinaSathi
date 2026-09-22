function metrics = evaluateReferableDR(truthGrades, predictedGrades)
%EVALUATEREFERABLEDR Binary referable threshold is grade >= 2.
truth = truthGrades(:) >= 2; predicted = predictedGrades(:) >= 2;
tp = sum(truth & predicted); tn = sum(~truth & ~predicted);
fp = sum(~truth & predicted); fn = sum(truth & ~predicted);
metrics = struct("sensitivity", tp/max(tp+fn, 1), "specificity", tn/max(tn+fp, 1), ...
    "positivePredictiveValue", tp/max(tp+fp, 1), "negativePredictiveValue", tn/max(tn+fn, 1));
end
