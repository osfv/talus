# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-164759_diffusion64\checkpoints\step_0030000.pt

step 30000, dataset `base64`/test, 1000 maps per half, guidance 2.0, 50 DDIM steps (quadratic spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1437 | 0.0495 | 2.90 |
| rapsd_distance | 0.2880 | 0.0103 | 28.04 |
| height_w1 | 0.0095 | 0.0084 | 1.13 |
| slope_w1_deg | 0.6078 | 0.2503 | 2.43 |

traversability pass rate: learned 0.902, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1611, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0019 | 0.017 | 1.000 | 1.116 |
| relief | 0.0084 | 0.074 | 0.995 | 1.148 |
| mean_slope_deg | 0.7570 | 0.111 | 0.991 | 1.081 |
| water_fraction | 0.0160 | 0.052 | 0.994 | 0.897 |
| spectral_beta | 0.7400 | 0.657 | 0.369 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.457 | 0.115 | 3.98 |
| hills | 0.385 | 0.129 | 2.99 |
| mountains | 0.321 | 0.131 | 2.45 |
| ridges | 0.257 | 0.102 | 2.52 |
| islands | 0.227 | 0.133 | 1.70 |
| mesas | 0.337 | 0.156 | 2.17 |

memorization: median NN-RMSE learned 0.0437 vs held-out 0.0432 (ratio 1.01); fraction of learned maps closer than held-out 1st percentile: 0.025

sampling: 179.3 ms/map

