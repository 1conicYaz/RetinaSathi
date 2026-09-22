function figureHandle = plot_simevents_dashboard(results, outputFile)
%PLOT_SIMEVENTS_DASHBOARD Create judge-readable scenario comparison charts.
arguments
    results = []
    outputFile (1,1) string = "../docs/images/retinasathi_simevents_results.png"
end
if isempty(results)
    results = readtable("results/simevents/scenarios_simevents.csv", "TextType", "string");
elseif isstruct(results)
    results = struct2table(results);
end

labels = replace(string(results.name), "_", " ");
figureHandle = figure("Name", "RetinaSathi SimEvents Results", "Color", "white", ...
    "Position", [80 80 1500 900]);
layout = tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");
title(layout, "RetinaSathi district workflow — verified SimEvents scenarios", ...
    "FontWeight", "bold", "FontSize", 16, "Color", "black");

nexttile;
bar([results.arrivals, results.patientsProcessedPerDay]);
title("Arrivals and completed screenings"); ylabel("Visits per simulated operating period");
legend("Arrivals", "Completed", "Location", "northwest");
localAxes(labels);

nexttile;
bar(results.maximumObservedTotalQueue, "FaceColor", [0.85 0.32 0.18]);
title("Maximum combined queue"); ylabel("Waiting visits");
localAxes(labels);

nexttile;
bar([results.averageWaitingMinutes, results.estimatedAverageTotalScreeningMinutes]);
title("Waiting and estimated total screening time"); ylabel("Minutes");
legend("Waiting", "Total screening", "Location", "northwest");
localAxes(labels);

nexttile;
bar(100*[results.cameraUtilization, results.aiUtilization, results.reviewerUtilization]);
title("Resource utilization"); ylabel("Utilization (%)"); ylim([0 105]);
legend("Camera", "AI station", "Reviewer", "Location", "northwest");
localAxes(labels);

annotation(figureHandle, "textbox", [0.02 0.005 0.96 0.035], ...
    "String", "Planning simulation based on declared assumptions; not observed clinical throughput.", ...
    "HorizontalAlignment", "center", "EdgeColor", "none", "FontAngle", "italic", ...
    "Color", "black");
set(findall(figureHandle, "Type", "axes"), "Color", "white", ...
    "XColor", "black", "YColor", "black", "GridColor", [0.78 0.78 0.78]);
set(findall(figureHandle, "Type", "text"), "Color", "black");
legends = findall(figureHandle, "Type", "legend");
set(legends, "Color", "white", "TextColor", "black", "EdgeColor", [0.5 0.5 0.5]);
if strlength(outputFile) > 0
    folder = fileparts(outputFile); if strlength(folder) > 0 && ~isfolder(folder), mkdir(folder); end
    exportgraphics(figureHandle, outputFile, "Resolution", 170);
end
end

function localAxes(labels)
grid on; box off;
xticks(1:numel(labels)); xticklabels(labels); xtickangle(25);
end
