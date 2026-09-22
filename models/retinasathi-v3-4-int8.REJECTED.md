# Rejected V3.4 INT8 experiment

Do not deploy `retinasathi-v3-4-int8.onnx`.

Dynamic INT8 quantization reduced the artifact from 84.27 MiB to about 24 MiB,
but the 25-case parity check found only 23/25 grade agreement with FP32 V3.4.
Two cases changed grade, and the maximum calibrated grade-probability shift was
0.1144. Referral agreement remained 25/25, but this is insufficient to call the
artifact equivalent to V3.4. The detailed result is in
`artifacts/v3_4_int8_parity.json`.
