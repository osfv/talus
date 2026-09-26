# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-164759_diffusion64\checkpoints\step_0020000.pt

step 20000, dataset `base64`/test, 1000 maps per half, guidance 1.5, 200 DDIM steps (uniform spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1577 | 0.0495 | 3.19 |
| rapsd_distance | 0.2362 | 0.0103 | 22.99 |
| height_w1 | 0.0095 | 0.0084 | 1.13 |
| slope_w1_deg | 0.7363 | 0.2503 | 2.94 |

traversability pass rate: learned 0.911, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1632, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0037 | 0.033 | 1.000 | 1.116 |
| relief | 0.0106 | 0.094 | 0.994 | 1.148 |
| mean_slope_deg | 0.9592 | 0.140 | 0.985 | 1.081 |
| water_fraction | 0.0158 | 0.052 | 0.995 | 0.897 |
| spectral_beta | 0.9917 | 0.880 | 0.269 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.403 | 0.115 | 3.51 |
| hills | 0.312 | 0.129 | 2.42 |
| mountains | 0.427 | 0.131 | 3.25 |
| ridges | 0.370 | 0.102 | 3.62 |
| islands | 0.276 | 0.133 | 2.07 |
| mesas | 0.343 | 0.156 | 2.21 |

memorization: median NN-RMSE learned 0.0434 vs held-out 0.0432 (ratio 1.00); fraction of learned maps closer than held-out 1st percentile: 0.031

sampling: 676.4 ms/map

