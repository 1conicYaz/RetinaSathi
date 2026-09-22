function [output, bounds] = removeBlackBorder(image)
%REMOVEBLACKBORDER Compatibility wrapper around cropFundus.
[output, bounds] = cropFundus(image);
end
