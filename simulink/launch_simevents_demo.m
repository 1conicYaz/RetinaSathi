function model = launch_simevents_demo(pacingRate, p)
%LAUNCH_SIMEVENTS_DEMO Open a paced, visible baseline workflow demonstration.
arguments
    pacingRate (1,1) double {mustBePositive} = 20
    p (1,1) struct = parameters()
end

model = build_retinasathi_workflow(p);
open_system(model);
set_param(model, "ZoomFactor", "FitSystem", "EnablePacing", "on", ...
    "PacingRate", string(pacingRate));
open_system(model + "/Capture Queue Scope");
open_system(model + "/AI Queue Scope");
open_system(model + "/Review Queue Scope");

fprintf("RetinaSathi SimEvents scenario '%s' is ready.\n", p.name);
fprintf("Pacing: %.1f simulation minutes per wall-clock second (about %.0f seconds total).\n", ...
    pacingRate, p.minutes/pacingRate);
fprintf("Press Run in Simulink, or execute: set_param(model,'SimulationCommand','start')\n");
fprintf("Watch Live Total Queue, Live Completed, and the three queue scopes.\n");
end
