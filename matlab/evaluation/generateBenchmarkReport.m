function report = generateBenchmarkReport(truth, predicted, outputPath)
%GENERATEBENCHMARKREPORT Save measured classifier metrics as JSON.
report = evaluateDR(truth, predicted);
report.referable = evaluateReferableDR(truth, predicted);
file = fopen(outputPath, "w"); cleanup = onCleanup(@() fclose(file));
fprintf(file, "%s", jsonencode(report, "PrettyPrint", true));
end
