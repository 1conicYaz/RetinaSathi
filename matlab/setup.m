function setup()
%SETUP Add the RetinaSathi MATLAB packages to the current MATLAB path.
root = fileparts(mfilename("fullpath"));
addpath(genpath(root));
fprintf("RetinaSathi MATLAB paths added from %s\n", root);
end
