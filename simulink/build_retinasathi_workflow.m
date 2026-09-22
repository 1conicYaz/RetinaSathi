function model = build_retinasathi_workflow(p)
%BUILD_RETINASATHI_WORKFLOW Build the executable RetinaSathi SimEvents model.
% One entity represents one retinal-screening visit. Use
% RUN_SIMEVENTS_WORKFLOW to simulate and collect operational metrics.

arguments
    p (1,1) struct = parameters()
end

localRequireProduct("Simulink", "Simulink");
localRequireProduct("SimEvents", "SimEvents");
load_system("sldelib");
load_system("simulink");

model = "RetinaSathiScreeningWorkflow";
modelDirectory = fileparts(mfilename("fullpath"));
modelFile = fullfile(modelDirectory, model + ".slx");
if bdIsLoaded(model), close_system(model, 0); end
if isfile(modelFile)
    load_system(modelFile);
    Simulink.BlockDiagram.deleteContents(model);
else
    new_system(model);
end
set_param(model, "StopTime", string(p.minutes), "SolverType", "Variable-step", ...
    "Solver", "VariableStepDiscrete", "ReturnWorkspaceOutputs", "on");

% Patient generation, capture, quality gate, and one recapture loop.
localAdd(model, "sldelib/Entity Generator", "Patient Arrivals", [30 235 145 305]);
arrivalAction = sprintf("persistent state; if isempty(state), state = %.0f; end; " + ...
    "state = mod(16807*state,2147483647); " + ...
    "u = max(double(state)/2147483647,1e-12); dt = -log(u)/%.15g;", ...
    p.seed, p.arrivalRatePerMinute);
qualityAction = sprintf("u = mod(double(entitySys.id)*7919 + %.0f,10000)/10000; " + ...
    "if u < %.15g, entity.route = 2; else, entity.route = 1; end;", ...
    p.seed, p.poorQualityRate);
set_param(model + "/Patient Arrivals", "TimeSource", "MATLAB action", ...
    "IntergenerationTimeAction", arrivalAction, "GenerateEntityAtSimulationStart", "off", ...
    "EntityType", "Structured", "AttributeName", "route", "AttributeInitialValue", "1", ...
    "GenerateAction", qualityAction, "NumberEntitiesDeparted", "on");

localAdd(model, "sldelib/Entity Input Switch", "Capture Merge", [185 235 255 325]);
set_param(model + "/Capture Merge", "NumberInputPorts", "2", "ActivePortSelection", "All");
localAdd(model, "sldelib/Entity Queue", "Capture Queue", [295 235 400 305]);
localConfigureQueue(model + "/Capture Queue", "FIFO", "", "Ascending");
localAdd(model, "sldelib/Entity Server", "Fundus Camera", [445 235 555 305]);
localConfigureServer(model + "/Fundus Camera", p.cameraCount, p.cameraMinutes, false, true);
localAdd(model, "sldelib/Entity Output Switch", "Quality Gate", [600 220 690 320]);
set_param(model + "/Quality Gate", "NumberOutputPorts", "2", ...
    "SwitchingCriterion", "From attribute", "SwitchAttributeName", "route");
localAdd(model, "sldelib/Entity Server", "Recapture", [600 385 710 455]);
localConfigureServer(model + "/Recapture", 1, p.cameraMinutes, true, false);
set_param(model + "/Recapture", "ServiceCompleteAction", "entity.route = 1;");

% Accepted image transfer and AI processing.
localAdd(model, "sldelib/Entity Server", "Network Transfer", [755 235 875 305]);
networkMinutes = effective_network_minutes(p);
localConfigureServer(model + "/Network Transfer", inf, networkMinutes, false, false);
localAdd(model, "sldelib/Entity Queue", "AI Queue", [920 235 1020 305]);
localConfigureQueue(model + "/AI Queue", "FIFO", "", "Ascending");
localAdd(model, "sldelib/Entity Server", "V3.4 AI Inference", [1065 235 1195 305]);
localConfigureServer(model + "/V3.4 AI Inference", p.aiDeviceCount, p.aiMinutes, false, true);
aiAction = sprintf("u1 = mod(double(entitySys.id)*3571 + %.0f,10000)/10000; " + ...
    "u2 = mod(double(entitySys.id)*6271 + %.0f,10000)/10000; " + ...
    "u3 = mod(double(entitySys.id)*4513 + %.0f,10000)/10000; " + ...
    "if u1 < %.15g, entity.route = 1; " + ...
    "if u3 < %.15g, entitySys.priority = 1; else, entitySys.priority = 2; end; " + ...
    "elseif u2 < %.15g, entity.route = 2; entitySys.priority = 3; " + ...
    "else, entity.route = 3; entitySys.priority = 4; end;", ...
    p.seed + 17, p.seed + 31, p.seed + 47, p.referableRate, ...
    p.severeRateWithinReferable, p.uncertainRate);
set_param(model + "/V3.4 AI Inference", "ServiceCompleteAction", aiAction);
localAdd(model, "sldelib/Entity Output Switch", "AI Decision Router", [1240 210 1335 330]);
set_param(model + "/AI Decision Router", "NumberOutputPorts", "3", ...
    "SwitchingCriterion", "From attribute", "SwitchAttributeName", "route");

% Review-required routes and priority clinical queue.
localAdd(model, "sldelib/Entity Server", "Referable Counter", [1390 115 1510 175]);
localConfigureServer(model + "/Referable Counter", inf, 0, true, false);
localAdd(model, "sldelib/Entity Server", "Uncertain Counter", [1390 235 1510 295]);
localConfigureServer(model + "/Uncertain Counter", inf, 0, true, false);
localAdd(model, "sldelib/Entity Input Switch", "Review Merge", [1555 145 1630 285]);
set_param(model + "/Review Merge", "NumberInputPorts", "2", "ActivePortSelection", "All");
localAdd(model, "sldelib/Entity Queue", "Clinical Review Queue", [1680 185 1810 255]);
if p.priorityReview
    localConfigureQueue(model + "/Clinical Review Queue", "Priority", "entitySys.priority", "Ascending");
else
    localConfigureQueue(model + "/Clinical Review Queue", "FIFO", "", "Ascending");
end
localAdd(model, "sldelib/Entity Server", "Human Review", [1860 185 1970 255]);
localConfigureServer(model + "/Human Review", p.reviewerCount, p.reviewMinutes, false, true);
localAdd(model, "sldelib/Entity Terminator", "Reviewed Outcome", [2020 185 2130 245]);
set_param(model + "/Reviewed Outcome", "NumberEntitiesArrived", "on");
localAdd(model, "sldelib/Entity Terminator", "Routine Outcome", [1390 365 1510 425]);
set_param(model + "/Routine Outcome", "NumberEntitiesArrived", "on");

% Entity paths. In R2026a, enabled statistics are placed before the entity
% output: Generator(d,entity), Queue(n,w,l,entity), Server(stat,entity).
localLine(model, "Patient Arrivals/2", "Capture Merge/1");
localLine(model, "Capture Merge/1", "Capture Queue/1");
localLine(model, "Capture Queue/4", "Fundus Camera/1");
localLine(model, "Fundus Camera/2", "Quality Gate/1");
localLine(model, "Quality Gate/1", "Network Transfer/1");
localLine(model, "Quality Gate/2", "Recapture/1");
localLine(model, "Recapture/2", "Capture Merge/2");
localLine(model, "Network Transfer/1", "AI Queue/1");
localLine(model, "AI Queue/4", "V3.4 AI Inference/1");
localLine(model, "V3.4 AI Inference/2", "AI Decision Router/1");
localLine(model, "AI Decision Router/1", "Referable Counter/1");
localLine(model, "AI Decision Router/2", "Uncertain Counter/1");
localLine(model, "AI Decision Router/3", "Routine Outcome/1");
localLine(model, "Referable Counter/2", "Review Merge/1");
localLine(model, "Uncertain Counter/2", "Review Merge/2");
localLine(model, "Review Merge/1", "Clinical Review Queue/1");
localLine(model, "Clinical Review Queue/4", "Human Review/1");
localLine(model, "Human Review/2", "Reviewed Outcome/1");

% Export live block statistics as timeseries.
metricY = 520;
localAddMetric(model, "Arrivals", "Patient Arrivals", 1, "se_arrivals", [40 metricY 145 metricY+28]);
localAddMetric(model, "Capture queue", "Capture Queue", 1, "se_capture_queue", [180 metricY 300 metricY+28]);
localAddMetric(model, "Capture wait", "Capture Queue", 2, "se_capture_wait", [330 metricY 450 metricY+28]);
localAddMetric(model, "Camera util", "Fundus Camera", 1, "se_camera_util", [480 metricY 590 metricY+28]);
localAddMetric(model, "Recaptures", "Recapture", 1, "se_recaptures", [620 metricY 730 metricY+28]);
localAddMetric(model, "AI queue", "AI Queue", 1, "se_ai_queue", [760 metricY 870 metricY+28]);
localAddMetric(model, "AI wait", "AI Queue", 2, "se_ai_wait", [900 metricY 1010 metricY+28]);
localAddMetric(model, "AI util", "V3.4 AI Inference", 1, "se_ai_util", [1040 metricY 1145 metricY+28]);
localAddMetric(model, "Referable", "Referable Counter", 1, "se_referable", [1175 metricY 1280 metricY+28]);
localAddMetric(model, "Uncertain", "Uncertain Counter", 1, "se_uncertain", [1310 metricY 1415 metricY+28]);
localAddMetric(model, "Review queue", "Clinical Review Queue", 1, "se_review_queue", [1445 metricY 1570 metricY+28]);
localAddMetric(model, "Review wait", "Clinical Review Queue", 2, "se_review_wait", [1600 metricY 1720 metricY+28]);
localAddMetric(model, "Reviewer util", "Human Review", 1, "se_reviewer_util", [1750 metricY 1870 metricY+28]);
localAddMetric(model, "Routine done", "Routine Outcome", 1, "se_routine_done", [1900 metricY 2010 metricY+28]);
localAddMetric(model, "Reviewed done", "Reviewed Outcome", 1, "se_reviewed_done", [2040 metricY 2160 metricY+28]);

% Visible live dashboard.
localAdd(model, "simulink/Math Operations/Add", "Total Queue", [1530 600 1570 680]);
set_param(model + "/Total Queue", "Inputs", "+++");
localLine(model, "Capture Queue/1", "Total Queue/1");
localLine(model, "AI Queue/1", "Total Queue/2");
localLine(model, "Clinical Review Queue/1", "Total Queue/3");
localAdd(model, "simulink/Sinks/Display", "Live Total Queue", [1610 610 1725 665]);
localLine(model, "Total Queue/1", "Live Total Queue/1");
localAdd(model, "simulink/Math Operations/Add", "Completed", [1870 600 1910 660]);
set_param(model + "/Completed", "Inputs", "++");
localLine(model, "Routine Outcome/1", "Completed/1");
localLine(model, "Reviewed Outcome/1", "Completed/2");
localAdd(model, "simulink/Sinks/Display", "Live Completed", [1950 600 2070 655]);
localLine(model, "Completed/1", "Live Completed/1");
% SimEvents statistics cannot be combined with a standard Mux. MathWorks
% requires one observation block for each signal.
localAdd(model, "simulink/Sinks/Scope", "Capture Queue Scope", [1050 610 1130 660]);
localLine(model, "Capture Queue/1", "Capture Queue Scope/1");
localAdd(model, "simulink/Sinks/Scope", "AI Queue Scope", [1170 610 1250 660]);
localLine(model, "AI Queue/1", "AI Queue Scope/1");
localAdd(model, "simulink/Sinks/Scope", "Review Queue Scope", [1290 610 1370 660]);
localLine(model, "Clinical Review Queue/1", "Review Queue Scope/1");

note = Simulink.Annotation(model, sprintf("RetinaSathi executable SimEvents workflow\n" + ...
    "Scenario: %s | %.0f min | seed %.0f\n" + ...
    "Network: %.2f min (%.2f MB at %.2f Mbps plus overhead)\n" + ...
    "Quality recapture, AI routing, priority review, outcomes, and live statistics\n" + ...
    "Planning assumptions only; this is not clinical evidence.", p.name, p.minutes, p.seed, ...
    networkMinutes, p.imageSizeMB, p.bandwidthMbps));
note.Position = [35 25 760 80]; note.FontSize = 13; note.FontWeight = "bold";
set_param(model + "/Quality Gate", "BackgroundColor", "orange");
set_param(model + "/AI Decision Router", "BackgroundColor", "lightBlue");
set_param(model + "/Clinical Review Queue", "BackgroundColor", "red");
set_param(model + "/Human Review", "BackgroundColor", "yellow");
set_param(model + "/Routine Outcome", "BackgroundColor", "green");
set_param(model + "/Reviewed Outcome", "BackgroundColor", "green");

Simulink.BlockDiagram.arrangeSystem(model, "FullLayout", "true");
save_system(model, modelFile);
end

function localRequireProduct(name, feature)
if ~license("test", feature)
    error("RetinaSathi:PRODUCT_UNAVAILABLE", "%s licence is unavailable.", name);
end
if name == "SimEvents" && isempty(ver("slde"))
    error("RetinaSathi:SIMEVENTS_NOT_INSTALLED", ...
        "SimEvents is licensed but not installed. Install it before building the workflow.");
end
end

function localAdd(model, source, name, position)
add_block(source, model + "/" + name, "Position", position);
end

function localConfigureQueue(path, queueType, prioritySource, direction)
set_param(path, "Capacity", "inf", "QueueType", queueType, ...
    "NumberEntitiesInBlock", "on", "AverageWait", "on", "AverageQueueLength", "on");
if queueType == "Priority"
    set_param(path, "PrioritySource", prioritySource, "SortingDirection", direction);
end
end

function localConfigureServer(path, capacity, minutes, exposeDepartures, exposeUtilization)
set_param(path, "Capacity", string(capacity), "ServiceTimeSource", "Dialog", ...
    "ServiceTimeValue", string(minutes), "NumberEntitiesDeparted", localOnOff(exposeDepartures), ...
    "Utilization", localOnOff(exposeUtilization));
end

function value = localOnOff(flag)
if flag, value = "on"; else, value = "off"; end
end

function localLine(model, source, destination)
add_line(model, source, destination, "autorouting", "on");
end

function localAddMetric(model, label, sourceBlock, sourcePort, variableName, position)
blockName = "Metric - " + label;
localAdd(model, "simulink/Sinks/To Workspace", blockName, position);
set_param(model + "/" + blockName, "VariableName", variableName, ...
    "SaveFormat", "Timeseries", "MaxDataPoints", "inf");
localLine(model, sourceBlock + "/" + sourcePort, blockName + "/1");
end
