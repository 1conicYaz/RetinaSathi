function metrics = run_simevents_workflow(p, outputDirectory)
%RUN_SIMEVENTS_WORKFLOW Build, simulate, summarize, and export one scenario.
arguments
    p (1,1) struct = parameters()
    outputDirectory (1,1) string = "results"
end

model = build_retinasathi_workflow(p);
cleanup = onCleanup(@() localClose(model));
simulation = sim(model, "StopTime", string(p.minutes));
series = struct("arrivals", simulation.se_arrivals, "captureQueue", simulation.se_capture_queue, ...
    "captureWait", simulation.se_capture_wait, "cameraUtil", simulation.se_camera_util, ...
    "recaptures", simulation.se_recaptures, "aiQueue", simulation.se_ai_queue, ...
    "aiWait", simulation.se_ai_wait, "aiUtil", simulation.se_ai_util, ...
    "referable", simulation.se_referable, "uncertain", simulation.se_uncertain, ...
    "reviewQueue", simulation.se_review_queue, "reviewWait", simulation.se_review_wait, ...
    "reviewerUtil", simulation.se_reviewer_util, "routineDone", simulation.se_routine_done, ...
    "reviewedDone", simulation.se_reviewed_done);

arrivals = localLast(series.arrivals); routineDone = localLast(series.routineDone);
reviewedDone = localLast(series.reviewedDone); processed = routineDone + reviewedDone;
captureWait = localLast(series.captureWait); aiWait = localLast(series.aiWait);
reviewWait = localLast(series.reviewWait); recaptures = localLast(series.recaptures);
referable = localLast(series.referable); uncertain = localLast(series.uncertain);
cameraUtil = localLast(series.cameraUtil); aiUtil = localLast(series.aiUtil);
reviewerUtil = localLast(series.reviewerUtil); networkMinutes = effective_network_minutes(p);
if processed > 0, reviewFraction = reviewedDone/processed; else, reviewFraction = 0; end
if arrivals > 0, recaptureFraction = recaptures/arrivals; else, recaptureFraction = 0; end
averageWaiting = captureWait + aiWait + reviewFraction*reviewWait;
averageTotal = averageWaiting + p.cameraMinutes*(1+recaptureFraction) + ...
    networkMinutes + p.aiMinutes + reviewFraction*p.reviewMinutes;
[bottleneck, bottleneckUtilization] = localBottleneck(cameraUtil, aiUtil, reviewerUtil);
metrics = struct("name", p.name, "engine", "SimEvents R2026a", "seed", p.seed, ...
    "minutes", p.minutes, "arrivals", arrivals, "patientsProcessedPerDay", processed, ...
    "annualThroughput", processed*p.workingDaysPerYear, ...
    "averageWaitingMinutes", averageWaiting, ...
    "estimatedAverageTotalScreeningMinutes", averageTotal, ...
    "captureAverageWaitMinutes", captureWait, ...
    "aiAverageWaitMinutes", aiWait, "reviewAverageWaitMinutes", reviewWait, ...
    "maximumCaptureQueue", localMax(series.captureQueue), "maximumAIQueue", localMax(series.aiQueue), ...
    "maximumReviewQueue", localMax(series.reviewQueue), ...
    "maximumObservedTotalQueue", localCombinedMax(series.captureQueue, series.aiQueue, series.reviewQueue), ...
    "cameraUtilization", cameraUtil, "aiUtilization", aiUtil, ...
    "reviewerUtilization", reviewerUtil, "bottleneck", bottleneck, ...
    "bottleneckUtilization", bottleneckUtilization, ...
    "imageSizeMB", p.imageSizeMB, "bandwidthMbps", p.bandwidthMbps, ...
    "networkDelayMinutesPerImage", networkMinutes, ...
    "networkDelayContributionMinutes", networkMinutes*(routineDone+referable+uncertain), ...
    "referableCount", referable, "uncertainCount", uncertain, ...
    "retakeCount", recaptures, "routineCompleted", routineDone, ...
    "reviewedCompleted", reviewedDone, "unfinished", max(arrivals-processed, 0));

if ~isfolder(outputDirectory), mkdir(outputDirectory); end
safeName = regexprep(p.name, "[^A-Za-z0-9_-]", "_");
writelines(jsonencode(metrics, "PrettyPrint", true), fullfile(outputDirectory, safeName+"_simevents.json"));
writetable(struct2table(metrics), fullfile(outputDirectory, safeName+"_simevents.csv"));
save(fullfile(outputDirectory, safeName+"_simevents.mat"), "metrics", "series", "p");
disp(struct2table(metrics));
clear cleanup
localClose(model)
end

function value = localLast(ts)
data = squeeze(ts.Data); if isempty(data), value = 0; else, value = double(data(end)); end
end

function value = localMax(ts)
data = squeeze(ts.Data); if isempty(data), value = 0; else, value = double(max(data, [], "all")); end
end

function value = localCombinedMax(varargin)
allTimes = [];
for index = 1:nargin, allTimes = [allTimes; varargin{index}.Time(:)]; end %#ok<AGROW>
allTimes = unique(allTimes); total = zeros(size(allTimes));
for index = 1:nargin
    ts = varargin{index}; if isempty(ts.Time), continue; end
    values = double(squeeze(ts.Data));
    [times, lastIndex] = unique(ts.Time(:), "last");
    values = values(lastIndex);
    if isscalar(times)
        contribution = repmat(values(1), size(allTimes));
    else
        contribution = interp1(times, values, allTimes, "previous", "extrap");
    end
    total = total + contribution;
end
if isempty(total), value = 0; else, value = max(total); end
end

function [name, utilization] = localBottleneck(camera, ai, reviewer)
[utilization, index] = max([camera, ai, reviewer]);
names = ["camera", "AI station", "human reviewer"];
name = names(index);
end

function localClose(model)
if bdIsLoaded(model), close_system(model, 0); end
end
