# MATLAB Toolbox Status

Audited by executing `check_toolboxes` on 2026-09-11 with MATLAB R2026a Update 5 (`26.1.0.3346908`) at `/Applications/MATLAB_R2026a.app/bin/matlab`.

| Component | Version | Licensed | Result |
|---|---:|---|---|
| MATLAB | 26.1 | yes | Executed |
| Image Processing Toolbox | 26.1 | yes | Executed by preprocessing/tests |
| Computer Vision Toolbox | 26.1 | yes | Available |
| Deep Learning Toolbox | 26.1 | yes | Executed by ONNX import/inference |
| Medical Imaging Toolbox | 26.1 | yes | Available; not required for fundus photographs |
| Statistics and Machine Learning Toolbox | 26.1 | yes | Available |
| Simulink | 26.1 | yes | Model built, saved, and simulated |
| SimEvents | 26.1 | yes | Executable entity workflow and all seven scenarios passed |

The functional preprocessing suite passed 3/3 tests. A licensed-local, non-official-test demo image completed preprocessing and baseline ONNX inference, and the app/report were rendered and visually inspected. Machine-local screenshots and `.mat` workspaces remain ignored because they contain licensed image derivatives.
