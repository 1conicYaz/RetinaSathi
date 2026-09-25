function [classNames, displayNames, colors] = lesionPalette()
%LESIONPALETTE Canonical class order, labels and colours for lesion evidence.
% These colours mark experimental candidate masks, not confirmed pathology.
classNames = ["microaneurysms","haemorrhages","hard_exudates","soft_exudates"];
displayNames = ["Microaneurysm candidates","Haemorrhage candidates", ...
    "Hard-exudate candidates","Soft-exudate candidates"];
colors = [1.00 .25 .21; ... % red
          1.00 .58 0.00; ... % orange
          1.00 .84 .04; ... % yellow
           .69 .32 .87];     % violet
end
