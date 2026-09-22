function [cropped, bounds] = cropFundus(image)
%CROPFUNDUS Remove near-black camera border using a robust intensity mask.
arguments
    image {mustBeNumeric}
end
gray = im2gray(image);
mask = gray > max(5, 0.03 * double(max(gray(:))));
rows = find(any(mask, 2));
columns = find(any(mask, 1));
if isempty(rows) || isempty(columns)
    cropped = image;
    bounds = [1 1 size(image, 2) size(image, 1)];
    return
end
padding = round(0.01 * min(size(gray)));
top = max(1, rows(1) - padding); bottom = min(size(gray, 1), rows(end) + padding);
left = max(1, columns(1) - padding); right = min(size(gray, 2), columns(end) + padding);
cropped = image(top:bottom, left:right, :);
bounds = [left top right-left+1 bottom-top+1];
end
