function window = launchRetinaSathiApp(config)
%LAUNCHRETINASATHIAPP Clinician-friendly RetinaSathi demonstration UI.
arguments
    config (1,1) struct = struct()
end
if ~isfield(config,"modelGeneration"), config.modelGeneration = "baseline"; end
if ~isfield(config,"visible"), config.visible = "on"; end
if string(config.modelGeneration) == "v3.4"
    if ~isfield(config,"modelPath"), config.modelPath = "../models/retinasathi-v3-4.onnx"; end
    if ~isfield(config,"manifestPath"), config.manifestPath = "../models/retinasathi-v3-4.manifest.json"; end
else
    if ~isfield(config,"modelPath"), config.modelPath = "../compute/artifacts/idrid_multitask.onnx"; end
end
if ~isfield(config,"lesionModelPath"), config.lesionModelPath = "../models/retinasathi-lesions-v3-2.onnx"; end
if ~isfield(config,"lesionManifestPath"), config.lesionManifestPath = "../models/retinasathi-lesions-v3-2.manifest.json"; end

c = struct("bg",[.965 .976 .973],"white",[1 1 1],"ink",[.055 .145 .125], ...
    "muted",[.35 .43 .41],"green",[.05 .42 .31],"greenSoft",[.88 .96 .92], ...
    "amber",[.80 .39 .05],"amberSoft",[1 .94 .84],"red",[.70 .17 .13], ...
    "redSoft",[1 .90 .88],"blue",[.12 .45 .73]);

window = uifigure("Name","RetinaSathi — Screening Support", ...
    "Position",[70 60 1320 820],"Color",c.bg,"Theme","light", ...
    "Visible",string(config.visible));
shell = uigridlayout(window,[3 1]);
shell.RowHeight = {70,"1x",38}; shell.Padding = [22 16 22 12]; shell.RowSpacing = 12;

header = uigridlayout(shell,[1 3]);
header.ColumnWidth = {"1x",250,175}; header.Padding = [4 0 4 0];
uilabel(header,"Text","RetinaSathi  •  MATLAB screening support  •  " + string(config.modelGeneration), ...
    "FontSize",22,"FontWeight","bold","FontColor",c.ink);
status = uilabel(header,"Text","Ready for a retinal image","HorizontalAlignment","right", ...
    "FontColor",c.muted,"FontSize",13); status.Layout.Column = 2;
chooseButton = uibutton(header,"push","Text","Choose retinal image","FontWeight","bold", ...
    "FontSize",13,"BackgroundColor",c.green,"FontColor",[1 1 1]);
chooseButton.Layout.Column = 3; chooseButton.ButtonPushedFcn = @analyze;

content = uigridlayout(shell,[1 2]);
content.ColumnWidth = {"3x","2x"}; content.ColumnSpacing = 14; content.Padding = [0 0 0 0];

imagePanel = uipanel(content,"Title","Retinal image","FontWeight","bold","FontSize",14, ...
    "BackgroundColor",c.white,"ForegroundColor",c.ink);
imageGrid = uigridlayout(imagePanel,[2 1]);
imageGrid.RowHeight = {"1x",46}; imageGrid.Padding = [10 8 10 10];
imageTabs = uitabgroup(imageGrid);
originalTab = uitab(imageTabs,"Title","Original photograph");
enhancedTab = uitab(imageTabs,"Title","Enhanced retinal view");
modelInputTab = uitab(imageTabs,"Title","Exact model input");
lesionTab = uitab(imageTabs,"Title","Lesion evidence");
originalImageGrid = uigridlayout(originalTab,[1 1]);
originalImageGrid.Padding = [8 8 8 8];
enhancedImageGrid = uigridlayout(enhancedTab,[1 1]);
enhancedImageGrid.Padding = [8 8 8 8];
modelInputGrid = uigridlayout(modelInputTab,[1 1]);
modelInputGrid.Padding = [8 8 8 8];
lesionImageGrid = uigridlayout(lesionTab,[3 1]);
lesionImageGrid.RowHeight = {"1x",40,54};
lesionImageGrid.RowSpacing = 6; lesionImageGrid.Padding = [8 8 8 8];
originalAxes = uiaxes(originalImageGrid);
enhancedAxes = uiaxes(enhancedImageGrid);
modelInputAxes = uiaxes(modelInputGrid);
lesionAxes = uiaxes(lesionImageGrid);
lesionLegend = uigridlayout(lesionImageGrid,[1 4]);
lesionLegend.Layout.Row = 2; lesionLegend.ColumnWidth = {"1x","1x","1x","1x"};
lesionLegend.ColumnSpacing = 6; lesionLegend.Padding = [0 0 0 0];
[~,legendLabels,legendColors] = lesionPalette();
for legendIndex = 1:numel(legendLabels)
    legendCard = uipanel(lesionLegend,"BorderType","line", ...
        "BackgroundColor",c.bg,"ForegroundColor",[.78 .83 .81]);
    legendItem = uigridlayout(legendCard,[1 2]);
    legendItem.ColumnWidth = {18,"1x"}; legendItem.Padding = [6 4 6 4];
    uilabel(legendItem,"Text","","BackgroundColor",legendColors(legendIndex,:));
    uilabel(legendItem,"Text",legendLabels(legendIndex),"FontSize",10, ...
        "FontWeight","bold","FontColor",c.ink,"WordWrap","on");
end
lesionHelp = uilabel(lesionImageGrid,"Text", ...
    "Experimental candidate masks will appear here after analysis.", ...
    "WordWrap","on","FontColor",c.muted,"FontSize",11);
lesionHelp.Layout.Row = 3;
configureImageAxes(originalAxes,c); configureImageAxes(enhancedAxes,c);
configureImageAxes(modelInputAxes,c);
configureImageAxes(lesionAxes,c);
imageHelp = uilabel(imageGrid,"Text", ...
    "Enhanced and exact-input tabs show preprocessing, not model attention, a heatmap or a lesion map.", ...
    "FontColor",c.muted,"FontSize",12,"WordWrap","on"); imageHelp.Layout.Row = 2;

resultPanel = uipanel(content,"Title","Screening result","FontWeight","bold","FontSize",14, ...
    "BackgroundColor",c.white,"ForegroundColor",c.ink);
resultGrid = uigridlayout(resultPanel,[5 1]);
resultGrid.RowHeight = {86,145,185,"1x",62};
resultGrid.RowSpacing = 10; resultGrid.Padding = [12 10 12 12];

decisionCard = uipanel(resultGrid,"BorderType","none","BackgroundColor",c.greenSoft);
decisionGrid = uigridlayout(decisionCard,[2 1]);
decisionGrid.RowHeight = {31,"1x"}; decisionGrid.Padding = [14 8 14 8];
decisionTitle = uilabel(decisionGrid,"Text","Choose an image to begin","FontSize",19, ...
    "FontWeight","bold","FontColor",c.ink);
decisionMessage = uilabel(decisionGrid,"Text","The result will appear here.", ...
    "FontSize",12,"FontColor",c.muted,"WordWrap","on");

metricsGrid = uigridlayout(resultGrid,[1 3]);
metricsGrid.ColumnWidth = {"1x","1x","1x"}; metricsGrid.ColumnSpacing = 8;
metricsGrid.Padding = [0 0 0 0];
[~,qualityValue,qualityCaption,~] = metricCard(metricsGrid,"IMAGE QUALITY",c);
[~,gradeValue,gradeCaption,~] = metricCard(metricsGrid,"SUGGESTED GRADE",c);
[~,referralValue,referralCaption,referralGrid] = metricCard(metricsGrid,"REFERRAL SCORE",c);
referralScaleAxes = uiaxes(referralGrid);
referralScaleAxes.Layout.Row = 4;
updateReferralScale(referralScaleAxes,0,20.7,c);

probPanel = uipanel(resultGrid,"Title","Five-grade model probabilities", ...
    "FontWeight","bold","BackgroundColor",c.white,"ForegroundColor",c.ink);
probGrid = uigridlayout(probPanel,[2 1]);
probGrid.RowHeight = {"1x",44}; probGrid.Padding = [8 3 8 6];
probAxes = uiaxes(probGrid);
probNote = uilabel(probGrid,"Text", ...
    sprintf("G0 No DR  •  G1 Mild  •  G2 Moderate  •  G3 Severe  •  G4 Proliferative\nTallest bar = exact-grade confidence, not overall accuracy."), ...
    "FontColor",c.muted,"FontSize",11,"WordWrap","on"); probNote.Layout.Row = 2;
resetChart(probAxes,c);

detailTabs = uitabgroup(resultGrid);
summaryTab = uitab(detailTabs,"Title","What this means");
technicalTab = uitab(detailTabs,"Title","Technical status");
futureXaiTab = uitab(detailTabs,"Title","Proposed explainability");
roadmapTab = uitab(detailTabs,"Title","Future model roadmap");
summaryGrid = uigridlayout(summaryTab,[2 1]);
summaryGrid.RowHeight = {"1x",43}; summaryGrid.Padding = [10 8 10 7];
plainText = uitextarea(summaryGrid,"Editable","off","Value","No result yet.","FontSize",12);
plainText.BackgroundColor = c.white; plainText.FontColor = c.ink;
confidenceHelp = uilabel(summaryGrid,"Text", ...
    "Referral score and exact-grade confidence answer different questions.", ...
    "FontColor",c.blue,"FontWeight","bold","WordWrap","on"); confidenceHelp.Layout.Row = 2;
technicalGrid = uigridlayout(technicalTab,[1 1]); technicalGrid.Padding = [8 8 8 8];
technicalTable = uitable(technicalGrid,"ColumnName",{"Module","Status"}, ...
    "ColumnEditable",[false false],"RowName",[]);
technicalTable.BackgroundColor = [c.white; c.bg]; technicalTable.ForegroundColor = c.ink;
technicalTable.ColumnWidth = {130,"auto"};
technicalTable.Data = ["Model","Waiting";"DME","Not assessed";"Lesions","Not available"; ...
    "Attention map","Not available";"Clinical review","Required"];
futureXaiGrid = uigridlayout(futureXaiTab,[2 1]);
futureXaiGrid.RowHeight = {34,"1x"}; futureXaiGrid.Padding = [10 8 10 8];
uilabel(futureXaiGrid,"Text","FUTURE GOAL — no patient-specific explanation is active", ...
    "FontWeight","bold","FontColor",c.amber,"BackgroundColor",c.amberSoft, ...
    "HorizontalAlignment","center");
futureXaiText = uitextarea(futureXaiGrid,"Editable","off","FontSize",12, ...
    "BackgroundColor",c.white,"FontColor",c.ink,"Value",[ ...
    "Proposed validation path:"; ...
    "1. Generate transformer attention rollout or gradient influence from the frozen V3.4 model."; ...
    "2. Test faithfulness by hiding highlighted regions and measuring whether the prediction changes."; ...
    "3. Test stability across small brightness, crop and camera variations."; ...
    "4. Compare attention with IDRiD lesion masks, while keeping attention separate from lesion segmentation."; ...
    "5. Ask ophthalmologists whether the explanation improves review without creating false confidence."; ...
    "6. Display a patient-specific map only after Python, ONNX and MATLAB outputs pass the frozen validation gate."]);
futureXaiText.Layout.Row = 2;
roadmapGrid = uigridlayout(roadmapTab,[2 1]);
roadmapGrid.RowHeight = {34,"1x"}; roadmapGrid.Padding = [10 8 10 8];
uilabel(roadmapGrid,"Text","FUTURE WORK — planned and not part of the current clinical evidence", ...
    "FontWeight","bold","FontColor",c.amber,"BackgroundColor",c.amberSoft, ...
    "HorizontalAlignment","center");
roadmapText = uitextarea(roadmapGrid,"Editable","off","FontSize",12, ...
    "BackgroundColor",c.white,"FontColor",c.ink,"Value",[ ...
    "1. Expand governed data across public DR datasets and de-identified Indian camera cohorts; split by patient, site and camera."; ...
    "2. Train quality/OOD, referral, ordinal grade, exact grade, DME-risk, lesion, vessel and landmark modules under separate validation gates."; ...
    "3. Improve segmentation with larger-context patches, centre-crop inference, anatomy-aware hard negatives and object-level false-positive metrics."; ...
    "4. Compare EfficientNet, DINO and retinal foundation teachers under one frozen protocol; use distillation only if a teacher proves better."; ...
    "5. Distil validated knowledge into a smaller edge student, then recalibrate and validate the student independently."; ...
    "6. Freeze ONNX artifacts and prove Python-to-MATLAB parity before deployment; use SimEvents to size cameras, compute, network and reviewers."; ...
    "7. Complete independent multi-camera and prospective ophthalmologist evaluation before any clinical-use claim."]);
roadmapText.Layout.Row = 2;

actionPanel = uipanel(resultGrid,"BorderType","none","BackgroundColor",c.amberSoft);
actionGrid = uigridlayout(actionPanel,[1 2]);
actionGrid.ColumnWidth = {105,"1x"}; actionGrid.Padding = [12 7 12 7];
uilabel(actionGrid,"Text","NEXT STEP","FontWeight","bold","FontColor",c.amber);
recommendationLabel = uilabel(actionGrid,"Text","Upload a retinal photograph.", ...
    "FontWeight","bold","FontColor",c.ink,"WordWrap","on");

uilabel(shell,"Text", ...
    "Research screening support only — not a diagnosis. A qualified clinician must confirm every result.", ...
    "HorizontalAlignment","center","FontSize",12,"FontColor",c.muted);

if isfield(config,"initialImagePath") && strlength(string(config.initialImagePath)) > 0
    drawnow; analyzePath(string(config.initialImagePath));
end

    function analyze(~,~)
        [name,folder] = uigetfile({"*.jpg;*.jpeg;*.png;*.tif;*.tiff","Fundus images"});
        if isequal(name,0), return; end
        analyzePath(string(fullfile(folder,name)));
    end

    function analyzePath(imagePath)
        try
            status.Text = "Analyzing image…"; chooseButton.Enable = "off"; drawnow;
            if string(config.modelGeneration) == "v3.4"
                result = runRetinaPipelineV34(imagePath,config);
                enhanced = result.preprocessing.enhanced;
            else
                result = runRetinaPipeline(imagePath,config); enhanced = result.enhanced;
            end
            imshow(result.original,"Parent",originalAxes);
            enhancedDisplay = enhanced;
            if isfield(result,"preprocessing")
                enhancedDisplay = cleanEnhancedDisplay(enhanced,result.preprocessing);
            end
            imshow(enhancedDisplay,"Parent",enhancedAxes);
            imshow(enhanced,"Parent",modelInputAxes);
            title(originalAxes,"Captured image");
            title(enhancedAxes,"Display-only enhanced retinal view");
            title(modelInputAxes,"Exact V3.4 model input — not a heatmap");
            if isfield(result,"lesions") && string(result.lesions.status)=="ready" && isfield(result.lesions,"overlay")
                imshow(result.lesions.overlay,"Parent",lesionAxes);
                title(lesionAxes,sprintf([ ...
                    'Experimental lesion candidate overlay\n' ...
                    'Colours identify candidate classes; clinician confirmation required']));
                lesionHelp.Text = lesionSummary(result.lesions);
            else
                cla(lesionAxes); configureImageAxes(lesionAxes,c);
                lesionHelp.Text = "Lesion evidence: " + friendlyStatus(moduleStatus(result,"lesions","not_trained"));
            end

            [gradeNumber,gradeName] = readableGrade(result.dr);
            isPoor = string(result.quality.label) == "poor";
            hasReferral = isfield(result.dr,"referable") && ~isempty(result.dr.referable);
            isReferable = hasReferral && logical(result.dr.referable);
            qualityValue.Text = sprintf("%.0f%%",100*result.quality.score);
            qualityCaption.Text = sentenceCase(string(result.quality.label));
            qualityValue.FontColor = c.green;
            if isPoor, qualityValue.FontColor = c.red; end
            if isPoor || isempty(gradeNumber)
                gradeValue.Text = "—"; gradeCaption.Text = "Not assessed";
            else
                gradeValue.Text = "G" + string(gradeNumber); gradeCaption.Text = gradeName;
            end

            scoreText = "—"; thresholdText = "";
            if isfield(result.dr,"referableScore") && ~isempty(result.dr.referableScore)
                scoreText = sprintf("%.1f%%",100*result.dr.referableScore);
            end
            if isfield(result.dr,"threshold") && ~isempty(result.dr.threshold)
                thresholdText = sprintf("Threshold %.1f%%",100*result.dr.threshold);
            end
            referralValue.Text = scoreText; referralCaption.Text = thresholdText;
            referralScorePercent = 0;
            thresholdPercent = 20.7;
            if isfield(result.dr,"referableScore") && ~isempty(result.dr.referableScore)
                referralScorePercent = 100*double(result.dr.referableScore);
            end
            if isfield(result.dr,"threshold") && ~isempty(result.dr.threshold)
                thresholdPercent = 100*double(result.dr.threshold);
            end
            updateReferralScale(referralScaleAxes,referralScorePercent,thresholdPercent,c);

            if isPoor
                setDecision("Image needs to be retaken", ...
                    "The photograph did not pass quality checks, so no clinical result is shown.",c.redSoft,c.red);
                recommendation = "Retake a clear, well-lit retinal photograph.";
            elseif isReferable
                setDecision("Referable diabetic retinopathy suspected", ...
                    "The model recommends specialist review. This screening result is not a diagnosis.",c.redSoft,c.red);
                recommendation = "Refer to an ophthalmologist for confirmatory examination.";
            elseif hasReferral
                setDecision("No referable DR detected", ...
                    "Continue routine care according to the clinical screening protocol.",c.greenSoft,c.green);
                recommendation = "Routine follow-up; seek earlier review if symptoms develop.";
            else
                setDecision("Result unavailable","The referral module did not return a usable result.",c.amberSoft,c.amber);
                recommendation = "Ask a clinician to review the retinal photograph.";
            end
            recommendationLabel.Text = recommendation;

            confidence = NaN;
            if isfield(result.dr,"confidence") && ~isempty(result.dr.confidence)
                confidence = double(result.dr.confidence);
            end
            if ~isnan(confidence) && ~isempty(gradeNumber)
                gradeCaption.Text = sprintf("%s · %.1f%%",gradeName,100*confidence);
            end
            updateChart(probAxes,result.dr,gradeNumber,c);
            plainText.Value = explanation(isPoor,isReferable,gradeName,confidence,scoreText,recommendation);

            technicalTable.Data = [ ...
                "DR classifier",friendlyStatus(string(result.dr.status)); ...
                "DME",friendlyStatus(moduleStatus(result,"dme","not_assessed")) + " — no V3.4 DME head"; ...
                "Lesion analysis",friendlyStatus(moduleStatus(result,"lesions","not_integrated_in_matlab")); ...
                "Vessel analysis",friendlyStatus(moduleStatus(result,"vessels","not_integrated_in_matlab")); ...
                "Optic disc",friendlyStatus(moduleStatus(result,"opticDisc","not_integrated_in_matlab")); ...
                "Fovea",friendlyStatus(moduleStatus(result,"fovea","not_integrated_in_matlab")); ...
                "Attention map",friendlyStatus(string(result.gradcam.status)) + " — MATLAB parity not validated"; ...
                "Human review","Pending — required for every result"];
            status.Text = "Analysis complete"; chooseButton.Text = "Analyze another image";
            chooseButton.Enable = "on";
        catch exception
            chooseButton.Enable = "on"; status.Text = "Analysis failed";
            status.Tooltip = exception.message;
            if string(window.Visible) == "on"
                uialert(window,exception.message,"RetinaSathi error");
            else
                rethrow(exception);
            end
        end
    end

    function setDecision(titleText,messageText,background,foreground)
        decisionCard.BackgroundColor = background; decisionTitle.Text = titleText;
        decisionTitle.FontColor = foreground; decisionMessage.Text = messageText;
        decisionMessage.FontColor = c.ink;
    end
end

function configureImageAxes(ax,c)
ax.XTick = []; ax.YTick = []; ax.Box = "off"; ax.Color = c.white;
title(ax,"No image selected","Color",c.muted,"FontWeight","normal");
end

function output = cleanEnhancedDisplay(enhanced,preprocessing)
% Hide square-padding artefacts for display only; inference remains unchanged.
output = enhanced;
if ~isstruct(preprocessing) || ~isfield(preprocessing,"resized"), return; end
resized = preprocessing.resized;
if size(resized,1) ~= size(enhanced,1) || size(resized,2) ~= size(enhanced,2), return; end
retinalField = max(resized,[],3) > 8;
outside = repmat(~retinalField,1,1,size(enhanced,3));
output(outside) = 0;
end

function [panel,valueLabel,captionLabel,grid] = metricCard(parent,heading,c)
panel = uipanel(parent,"BorderType","line","BackgroundColor",c.bg,"ForegroundColor",[.80 .85 .83]);
grid = uigridlayout(panel,[4 1]);
grid.RowHeight = {20,34,22,"1x"}; grid.Padding = [10 7 10 5]; grid.RowSpacing = 1;
uilabel(grid,"Text",heading,"FontSize",10,"FontWeight","bold","FontColor",c.muted);
valueLabel = uilabel(grid,"Text","—","FontSize",24,"FontWeight","bold","FontColor",c.ink);
captionLabel = uilabel(grid,"Text","Waiting","FontSize",11,"FontColor",c.muted,"WordWrap","on");
end

function updateReferralScale(ax,score,threshold,c)
% Draw a compact, dependency-free referral scale with a visible threshold.
cla(ax); hold(ax,"on");
score = min(max(double(score),0),100);
threshold = min(max(double(threshold),0),100);
patch(ax,[0 threshold threshold 0],[0 0 1 1],c.greenSoft, ...
    "EdgeColor","none","FaceAlpha",1);
patch(ax,[threshold 100 100 threshold],[0 0 1 1],c.redSoft, ...
    "EdgeColor","none","FaceAlpha",1);
plot(ax,[threshold threshold],[0 1],"--","Color",c.red,"LineWidth",1.3);
plot(ax,score,.5,"v","MarkerSize",8,"MarkerFaceColor",c.blue, ...
    "MarkerEdgeColor",c.blue);
text(ax,threshold,.06,"threshold","HorizontalAlignment","center", ...
    "VerticalAlignment","bottom","FontSize",8,"Color",c.red);
ax.XLim = [0 100]; ax.YLim = [0 1]; ax.YTick = [];
ax.XTick = [0 50 100]; ax.XTickLabel = {"0","50","100%"};
ax.Box = "on"; ax.FontSize = 8; ax.Color = c.white;
ax.Toolbar.Visible = "off";
hold(ax,"off");
end

function resetChart(ax,c)
bar(ax,0:4,zeros(1,5),.62,"FaceColor",c.blue);
ax.XTick = 0:4; ax.XTickLabel = {"G0","G1","G2","G3","G4"};
ax.YLim = [0 100]; ylabel(ax,"%"); ax.Box = "off"; ax.Color = c.white; grid(ax,"on");
end

function updateChart(ax,dr,gradeNumber,c)
cla(ax);
if ~isfield(dr,"gradeProbabilities") || numel(dr.gradeProbabilities) ~= 5
    resetChart(ax,c); title(ax,"Probabilities unavailable","FontWeight","normal"); return;
end
values = 100*double(dr.gradeProbabilities(:))';
bars = bar(ax,0:4,values,.62,"FaceColor","flat");
bars.CData = repmat([.60 .70 .68],5,1);
if ~isempty(gradeNumber) && gradeNumber >= 0 && gradeNumber <= 4
    bars.CData(gradeNumber+1,:) = c.blue;
end
ax.XTick = 0:4; ax.XTickLabel = {"G0","G1","G2","G3","G4"};
ax.YLim = [0 110]; ylabel(ax,"%"); ax.Box = "off"; ax.Color = c.white; grid(ax,"on");
text(ax,0:4,values+4,compose("%.1f%%",values),"HorizontalAlignment","center","FontSize",9,"Color",c.ink);
end

function lines = explanation(isPoor,isReferable,gradeName,confidence,scoreText,recommendation)
if isPoor
    lines = ["The image quality is not sufficient for reliable screening."; ...
        "No DR grade or referral result should be used from this image."; ...
        "Next step: " + recommendation]; return;
end
if isnan(confidence)
    confidenceSentence = "The exact-grade confidence is unavailable.";
elseif confidence < .60
    confidenceSentence = sprintf("The exact grade is uncertain (%.1f%%); nearby grades also received substantial probability.",100*confidence);
else
    confidenceSentence = sprintf("The highest exact-grade probability is %.1f%%.",100*confidence);
end
if isReferable
    referralSentence = "The separate referral model recommends specialist review (score " + scoreText + ").";
else
    referralSentence = "The separate referral model did not cross the referral threshold (score " + scoreText + ").";
end
lines = ["Suggested severity: " + gradeName + "."; referralSentence; confidenceSentence; ...
    "Why scores differ: referral asks ‘specialist review or routine?’, while grade confidence asks ‘which exact grade?’"; ...
    "Next step: " + recommendation];
end

function [gradeNumber,gradeName] = readableGrade(dr)
gradeNumber = []; gradeName = "Not assessed";
if ~isfield(dr,"grade") || isempty(dr.grade), return; end
gradeNumber = double(dr.grade);
names = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
if gradeNumber >= 0 && gradeNumber <= 4 && gradeNumber == floor(gradeNumber)
    gradeName = names(gradeNumber+1);
else
    gradeName = "Unknown grade";
end
end

function output = sentenceCase(input)
output = replace(string(input),"_"," ");
if strlength(output)>0, output = upper(extractBefore(output,2)) + extractAfter(output,1); end
end

function output = friendlyStatus(input)
value = lower(string(input));
switch value
    case {"ready","candidate","available"}
        output = "Available";
    case {"not_assessed","not_integrated_in_matlab","not_trained","unavailable"}
        output = "Not available in this version";
    case {"ungradeable","retake_required"}
        output = "Stopped — image must be retaken";
    otherwise
        output = sentenceCase(value);
end
end

function status = moduleStatus(result,field,fallback)
if isfield(result,field) && isstruct(result.(field)) && isfield(result.(field),"status")
    status = string(result.(field).status);
else
    status = string(fallback);
end
end

function output = lesionSummary(lesions)
if ~isfield(lesions,"regions") || isempty(lesions.regions)
    output = "No candidate regions crossed the class-specific thresholds. This does not rule out disease.";
    return
end
variableNames = string(lesions.regions.Properties.VariableNames);
if any(variableNames=="Type")
    types = string(lesions.regions.Type);
elseif any(variableNames=="Class")
    % Accept the earlier prototype schema so saved results remain readable.
    types = string(lesions.regions.Class);
elseif width(lesions.regions)>=1
    % A defensive fallback keeps the clinical summary usable if an older
    % MATLAB release assigned Var1... names to the otherwise valid table.
    types = string(lesions.regions{:,1});
else
    output = "Lesion candidates were produced, but their class labels could not be displayed. Human review is still required.";
    return
end
[names,labels] = lesionPalette();
labels = replace(labels," candidates","");
parts = strings(0);
for index=1:numel(names)
    count=sum(types==names(index));
    if count>0
        countText=string(count);
        if count>=50, countText="50+"; end
        parts(end+1)=labels(index)+": "+countText; %#ok<AGROW>
    end
end
metric="";
if isfield(lesions,"validationMeanDice") && isfinite(lesions.validationMeanDice)
    metric=sprintf(" Internal full-image mean Dice %.3f.",lesions.validationMeanDice);
end
output = "Experimental " + string(lesions.modelVersion) + "  •  displayed candidate regions: " + strjoin(parts,"  •  ") + "." + metric + " Counts are not lesion burden. Candidate masks are not diagnoses; border and optic-disc false positives remain.";
end
