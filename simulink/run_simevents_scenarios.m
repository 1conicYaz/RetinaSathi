function results = run_simevents_scenarios(outputDirectory)
%RUN_SIMEVENTS_SCENARIOS Execute all seven scenarios in the SimEvents model.
arguments, outputDirectory (1,1) string = "results/simevents", end
base = parameters(); scenarios = repmat(base, 1, 7);
names = ["baseline_rural_clinic", "network_constrained", "increased_patient_load", ...
    "additional_camera", "additional_reviewer", "priority_review", "district_100k_annual"];
for index = 1:7, scenarios(index).name = names(index); end
scenarios(2).bandwidthMbps = 0.10;
scenarios(3).arrivalRatePerMinute = 0.22;
scenarios(4).arrivalRatePerMinute = 0.22; scenarios(4).cameraCount = 2;
scenarios(5).arrivalRatePerMinute = 0.22; scenarios(5).reviewerCount = 2;
scenarios(6).arrivalRatePerMinute = 0.22; scenarios(6).cameraCount = 2; scenarios(6).reviewMinutes = 12;
scenarios(6).referableRate = 0.55; scenarios(6).uncertainRate = 0.15; scenarios(6).priorityReview = true;
scenarios(7).minutes = 600; scenarios(7).arrivalRatePerMinute = 500/600; scenarios(7).cameraCount = 6;
scenarios(7).aiDeviceCount = 2; scenarios(7).reviewerCount = 4; scenarios(7).seed = 26044;
if ~isfolder(outputDirectory), mkdir(outputDirectory); end
for index = 1:numel(scenarios)
    results(index) = run_simevents_workflow(scenarios(index), outputDirectory); %#ok<AGROW>
end
writelines(jsonencode(results, "PrettyPrint", true), fullfile(outputDirectory, "scenarios_simevents.json"));
writetable(struct2table(results), fullfile(outputDirectory, "scenarios_simevents.csv"));
% Leave the checked-in judge model in the understandable baseline state.
build_retinasathi_workflow(base);
end
