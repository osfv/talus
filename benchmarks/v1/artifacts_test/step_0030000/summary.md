# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-164759_diffusion64\checkpoints\step_0030000.pt

step 30000, dataset `base64`/test, 1000 maps per half, guidance 1.5, 200 DDIM steps (uniform spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1486 | 0.0495 | 3.00 |
| rapsd_distance | 0.2466 | 0.0103 | 24.00 |
| height_w1 | 0.0091 | 0.0084 | 1.09 |
| slope_w1_deg | 0.7167 | 0.2503 | 2.86 |

traversability pass rate: learned 0.907, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1605, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0018 | 0.016 | 1.000 | 1.116 |
| relief | 0.0085 | 0.075 | 0.995 | 1.148 |
| mean_slope_deg | 0.7955 | 0.116 | 0.991 | 1.081 |
| water_fraction | 0.0157 | 0.051 | 0.994 | 0.897 |
| spectral_beta | 0.7757 | 0.689 | 0.363 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.431 | 0.115 | 3.75 |
| hills | 0.384 | 0.129 | 2.98 |
| mountains | 0.358 | 0.131 | 2.73 |
| ridges | 0.285 | 0.102 | 2.78 |
| islands | 0.229 | 0.133 | 1.72 |
| mesas | 0.319 | 0.156 | 2.05 |

memorization: median NN-RMSE learned 0.0438 vs held-out 0.0432 (ratio 1.01); fraction of learned maps closer than held-out 1st percentile: 0.024

sampling: 723.7 ms/map

