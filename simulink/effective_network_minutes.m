function minutes = effective_network_minutes(p)
%EFFECTIVE_NETWORK_MINUTES Convert image size and bandwidth to transfer time.
if isfield(p, "useBandwidthModel") && p.useBandwidthModel
    argumentsToCheck = [p.imageSizeMB, p.bandwidthMbps, p.networkOverheadMinutes];
    if any(~isfinite(argumentsToCheck)) || p.imageSizeMB < 0 || ...
            p.bandwidthMbps <= 0 || p.networkOverheadMinutes < 0
        error("RetinaSathi:INVALID_NETWORK_PARAMETERS", ...
            "Image size/overhead must be nonnegative and bandwidth must be positive.");
    end
    minutes = p.networkOverheadMinutes + (p.imageSizeMB * 8) / (p.bandwidthMbps * 60);
else
    minutes = p.networkMinutes;
end
end
