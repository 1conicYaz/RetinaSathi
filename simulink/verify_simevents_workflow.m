function report = verify_simevents_workflow(runAllScenarios, outputDirectory)
%VERIFY_SIMEVENTS_WORKFLOW Run software and scenario acceptance checks.
% This verifies implementation behavior, not clinical validity.
arguments
    runAllScenarios (1,1) logical = true
    outputDirectory (1,1) string = "results/simevents"
end

testName = strings(0,1); passed = false(0,1); evidence = strings(0,1);
append("SimEvents installed", ~isempty(ver("slde")), string(ver("slde").Version));
append("SimEvents licence", logical(license("test", "SimEvents")), ...
    "license('test','SimEvents') must return 1");

p = parameters();
model = build_retinasathi_workflow(p);
required = ["Patient Arrivals", "Capture Queue", "Fundus Camera", "Quality Gate", ...
    "Recapture", "Network Transfer", "AI Queue", "V3.4 AI Inference", ...
    "AI Decision Router", "Clinical Review Queue", "Human Review", ...
    "Routine Outcome", "Reviewed Outcome"];
blocks = string(get_param(find_system(model, "SearchDepth", 1, "Type", "Block"), "Name"));
append("Required blocks", all(ismember(required, blocks)), ...
    sprintf("%d/%d required blocks found", sum(ismember(required, blocks)), numel(required)));
close_system(model, 0);

baseline = run_simevents_workflow(p, outputDirectory);
append("Baseline produces visits", baseline.arrivals > 0 && baseline.patientsProcessedPerDay > 0, ...
    sprintf("%g arrivals; %g completed", baseline.arrivals, baseline.patientsProcessedPerDay));
append("Outcome accounting", baseline.routineCompleted + baseline.reviewedCompleted == ...
    baseline.patientsProcessedPerDay, "routine + reviewed equals completed");
append("Utilization bounds", all([baseline.cameraUtilization, baseline.aiUtilization, ...
    baseline.reviewerUtilization] >= 0 & [baseline.cameraUtilization, ...
    baseline.aiUtilization, baseline.reviewerUtilization] <= 1), "all utilization values are 0–1");
append("Bandwidth calculation", abs(baseline.networkDelayMinutesPerImage - 1.5) < 1e-9, ...
    sprintf("%.3f minutes/image", baseline.networkDelayMinutesPerImage));
append("Bottleneck generated", strlength(baseline.bottleneck) > 0, ...
    baseline.bottleneck + " at " + compose("%.1f%%",100*baseline.bottleneckUtilization));
append("Total screening time", baseline.estimatedAverageTotalScreeningMinutes > ...
    baseline.averageWaitingMinutes, compose("%.2f minutes",baseline.estimatedAverageTotalScreeningMinutes));

if runAllScenarios
    scenarios = run_simevents_scenarios(outputDirectory);
else
    scenarios = table2struct(readtable(fullfile(outputDirectory, "scenarios_simevents.csv"), ...
        "TextType", "string"));
end
append("Seven scenarios", numel(scenarios) == 7, sprintf("%d scenarios available",numel(scenarios)));
names = string({scenarios.name});
busy = scenarios(names == "increased_patient_load");
camera = scenarios(names == "additional_camera");
network = scenarios(names == "network_constrained");
district = scenarios(names == "district_100k_annual");
append("Network constraint changes delay", network.networkDelayMinutesPerImage > ...
    baseline.networkDelayMinutesPerImage, compose("%.2f vs %.2f min", ...
    network.networkDelayMinutesPerImage, baseline.networkDelayMinutesPerImage));
append("Extra camera improves throughput", camera.patientsProcessedPerDay > ...
    busy.patientsProcessedPerDay, sprintf("%g vs %g completed", ...
    camera.patientsProcessedPerDay, busy.patientsProcessedPerDay));
append("Extra camera reduces queue", camera.maximumObservedTotalQueue < ...
    busy.maximumObservedTotalQueue, sprintf("%g vs %g waiting", ...
    camera.maximumObservedTotalQueue, busy.maximumObservedTotalQueue));
append("District exceeds 100k planning target", district.annualThroughput >= 100000, ...
    sprintf("%g annual-equivalent screenings",district.annualThroughput));

report = table(testName(:), passed(:), evidence(:), ...
    'VariableNames', {'test', 'passed', 'evidence'});
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
writetable(report, fullfile(outputDirectory, "verification_report.csv"));
writelines(jsonencode(table2struct(report), "PrettyPrint", true), ...
    fullfile(outputDirectory, "verification_report.json"));
lines = ["# RetinaSathi SimEvents verification", "", ...
    "Generated: " + string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss Z")), "", ...
    "> This verifies software behavior and configured planning scenarios. It does not prove clinical safety.", "", ...
    "| Check | Result | Evidence |", "|---|---|---|"];
for index = 1:height(report)
    result = "FAIL"; if report.passed(index), result = "PASS"; end
    lines(end+1) = "| " + report.test(index) + " | " + result + " | " + ...
        replace(report.evidence(index), "|", "/") + " |"; %#ok<AGROW>
end
writelines(lines, fullfile(outputDirectory, "VERIFICATION_REPORT.md"));
disp(report);
if ~all(report.passed)
    error("RetinaSathi:SIMEVENTS_VERIFICATION_FAILED", ...
        "%d SimEvents verification checks failed.", sum(~report.passed));
end
fprintf("PASS: %d/%d SimEvents software checks.\n",sum(report.passed),height(report));

    function append(name, condition, detail)
        condition = all(logical(condition), "all");
        detail = string(detail); detail = detail(1);
        testName(end+1,1) = string(name);
        passed(end+1,1) = condition;
        evidence(end+1,1) = detail;
    end
end
