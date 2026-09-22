function output = enhanceFundus(image)
%ENHANCEFUNDUS Apply conservative luminance CLAHE without changing hue labels.
rgb = im2uint8(image);
lab = rgb2lab(rgb);
luminance = lab(:, :, 1) / 100;
lab(:, :, 1) = 100 * adapthisteq(luminance, "ClipLimit", 0.01, "NumTiles", [8 8]);
output = im2uint8(lab2rgb(lab));
end
