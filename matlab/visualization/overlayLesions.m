function output = overlayLesions(image, masks, colors)
%OVERLAYLESIONS Overlay genuine segmentation masks, never attention maps.
arguments
    image
    masks
    colors = lines(size(masks, 3))
end
output = im2double(image);
for index = 1:size(masks, 3)
    mask = logical(masks(:, :, index));
    for channel = 1:3
        layer = output(:, :, channel);
        layer(mask) = 0.55*layer(mask) + 0.45*colors(index, channel);
        output(:, :, channel) = layer;
    end
end
end
