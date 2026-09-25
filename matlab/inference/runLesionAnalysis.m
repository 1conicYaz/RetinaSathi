function result = runLesionAnalysis(image, modelPath, manifestPath)
%RUNLESIONANALYSIS Run an explicitly experimental four-class lesion model.
% Output masks are anatomical candidates, not confirmed clinical findings.
arguments
    image {mustBeNumeric}
    modelPath (1,1) string
    manifestPath (1,1) string = ""
end
if strlength(modelPath)==0 || ~isfile(modelPath)
    result = unavailable("not_trained", "No validated lesion ONNX was configured."); return
end
if strlength(manifestPath)==0
    manifestPath = replace(modelPath,".onnx",".manifest.json");
end
if ~isfile(manifestPath)
    result = unavailable("unavailable", "Lesion manifest was not found."); return
end
persistent net loadedPath
try
    manifest = jsondecode(fileread(manifestPath));
    if isfield(manifest,"release_gate") && isfield(manifest.release_gate,"passed") && ~manifest.release_gate.passed
        result = unavailable("failed_validation", "Lesion model did not pass its frozen validation gate."); return
    end
    if isfield(manifest,"onnx_sha256")
        actual = sha256File(modelPath);
        if actual ~= lower(string(manifest.onnx_sha256))
            error("RetinaSathi:ArtifactIntegrity","Lesion ONNX SHA-256 mismatch.");
        end
    end
    classes = string(manifest.classes(:)); tileSize = double(manifest.input_size);
    overlap = 128; minimumPixels = 3; sigmaScale = .125; borderMarginFraction = .015;
    if isfield(manifest,"inference")
        if isfield(manifest.inference,"overlap"), overlap=double(manifest.inference.overlap); end
        if isfield(manifest.inference,"minimum_region_pixels"), minimumPixels=double(manifest.inference.minimum_region_pixels); end
        if isfield(manifest.inference,"gaussian_sigma_scale"), sigmaScale=double(manifest.inference.gaussian_sigma_scale); end
        if isfield(manifest.inference,"retinal_border_margin_fraction"), borderMarginFraction=double(manifest.inference.retinal_border_margin_fraction); end
    end
    thresholds = readThresholds(manifest,classes);
    if isempty(net) || loadedPath ~= modelPath
        net = importNetworkFromONNX(modelPath,"InputDataFormats","BCSS","OutputDataFormats","BCSS");
        loadedPath = modelPath;
    end
    rgb = im2uint8(image); if ismatrix(rgb), rgb=repmat(rgb,1,1,3); end
    originalHeight=size(rgb,1); originalWidth=size(rgb,2);
    padHeight=max(0,tileSize-originalHeight); padWidth=max(0,tileSize-originalWidth);
    work=padarray(rgb,[padHeight padWidth],"symmetric","post");
    rows=tileStarts(size(work,1),tileSize,overlap); columns=tileStarts(size(work,2),tileSize,overlap);
    summed=zeros(size(work,1),size(work,2),numel(classes),"single"); weights=zeros(size(work,1),size(work,2),"single");
    importance=gaussianImportanceMap(tileSize,sigmaScale);
    meanRGB=reshape(single([.485 .456 .406]),1,1,3); stdRGB=reshape(single([.229 .224 .225]),1,1,3);
    for top=rows
        for left=columns
            tile=work(top:top+tileSize-1,left:left+tileSize-1,:);
            normalized=(single(tile)/255-meanRGB)./stdRGB;
            prediction=extractdata(predict(net,reshape(normalized,tileSize,tileSize,3,1)));
            logits=toHWC(prediction,numel(classes),tileSize);
            summed(top:top+tileSize-1,left:left+tileSize-1,:)=summed(top:top+tileSize-1,left:left+tileSize-1,:)+logits.*importance;
            weights(top:top+tileSize-1,left:left+tileSize-1)=weights(top:top+tileSize-1,left:left+tileSize-1)+importance;
        end
    end
    probabilities=1./(1+exp(-summed./max(weights,1)));
    probabilities=probabilities(1:originalHeight,1:originalWidth,:);
    retinalField=max(rgb,[],3)>12;
    marginPixels=max(1,round(min(originalHeight,originalWidth)*borderMarginFraction));
    retinalField=bwdist(~retinalField)>marginPixels;
    probabilities=repmat(single(retinalField),1,1,numel(classes)).*probabilities;
    masks=false(size(probabilities)); regions=table();
    for index=1:numel(classes)
        masks(:,:,index)=probabilities(:,:,index)>=thresholds(index);
        connected=bwconncomp(masks(:,:,index),4);
        stats=regionprops(connected,probabilities(:,:,index),"Area","BoundingBox","Centroid","MeanIntensity","MaxIntensity");
        [~,regionOrder]=sort([stats.Area],"descend");
        regionOrder=regionOrder(1:min(50,numel(regionOrder)));
        for regionIndex=regionOrder
            if stats(regionIndex).Area<minimumPixels, continue; end
            % Use the character-vector name/value form here. In MATLAB R2026a,
            % string scalars can be interpreted as additional table data,
            % producing Var1...Var8 instead of the declared schema.
            row=table(classes(index),stats(regionIndex).Area,{stats(regionIndex).BoundingBox},{stats(regionIndex).Centroid},stats(regionIndex).MeanIntensity,stats(regionIndex).MaxIntensity, ...
                'VariableNames',{'Type','AreaPixels','BoundingBox','Centroid','MeanProbability','MaxProbability'});
            if width(regions)==0, regions=row; else, regions=[regions;row]; end %#ok<AGROW>
        end
    end
    [paletteClasses,~,colors]=lesionPalette();
    if ~isequal(classes(:)',paletteClasses)
        error("RetinaSathi:LesionClassOrder", ...
            "Lesion manifest class order does not match the display palette.");
    end
    validationMeanDice=NaN; validationScope="not_recorded";
    if isfield(manifest,"validation") && isfield(manifest.validation,"mean_dice"), validationMeanDice=double(manifest.validation.mean_dice); end
    if isfield(manifest,"validation_scope"), validationScope=string(manifest.validation_scope); end
    result=struct("status","ready","assessmentState","experimental_lesion_evidence", ...
        "experimental",true,"modelVersion",string(manifest.model_version),"classes",classes, ...
        "thresholds",thresholds,"probabilities",probabilities,"masks",masks,"regions",regions, ...
        "validationMeanDice",validationMeanDice,"validationScope",validationScope, ...
        "overlay",overlayLesions(rgb,masks,colors), ...
        "disclaimer","Experimental candidates only. Border and optic-disc false positives remain; ophthalmologist confirmation is required.");
catch exception
    result=unavailable("error",exception.message);
end
end

function weights=gaussianImportanceMap(tileSize,sigmaScale)
coordinates=single(0:tileSize-1)-single(tileSize-1)/2;
sigma=single(tileSize*sigmaScale);
axisWeights=exp(-.5*(coordinates./sigma).^2);
weights=single(axisWeights(:)*axisWeights(:)');
weights=max(weights./max(weights,[],"all"),single(1e-3));
end

function output=unavailable(status,reason)
output=struct("status",status,"experimental",true,"reason",reason,"masks",[],"regions",table(),"overlay",[]);
end

function values=readThresholds(manifest,classes)
values=.5*ones(numel(classes),1);
if isfield(manifest,"thresholds")
    if isstruct(manifest.thresholds)
        for index=1:numel(classes)
            field=matlab.lang.makeValidName(classes(index));
            if isfield(manifest.thresholds,field)
                values(index)=double(manifest.thresholds.(field));
            end
        end
    elseif isnumeric(manifest.thresholds)
        values=double(manifest.thresholds(:));
    end
elseif isfield(manifest,"threshold")
    values(:)=double(manifest.threshold);
end
end

function starts=tileStarts(lengthValue,tileSize,overlap)
if lengthValue<=tileSize, starts=1; return; end
starts=1:(tileSize-overlap):(lengthValue-tileSize+1);
last=lengthValue-tileSize+1; if starts(end)~=last, starts=[starts last]; end
end

function output=toHWC(value,classCount,tileSize)
value=squeeze(value);
if isequal(size(value),[tileSize tileSize classCount]), output=single(value); return; end
if isequal(size(value),[classCount tileSize tileSize]), output=permute(single(value),[2 3 1]); return; end
error("RetinaSathi:LesionShape","Unexpected lesion ONNX output shape: %s",mat2str(size(value)));
end
