# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-192102_diffusion64_cont\checkpoints\step_0035000.pt

step 35000, dataset `base64`/test, 1000 maps per half, guidance 1.5, 200 DDIM steps (uniform spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1589 | 0.0495 | 3.21 |
| rapsd_distance | 0.2287 | 0.0103 | 22.26 |
| height_w1 | 0.0091 | 0.0084 | 1.08 |
| slope_w1_deg | 0.7550 | 0.2503 | 3.02 |

traversability pass rate: learned 0.912, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1601, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0015 | 0.014 | 1.000 | 1.116 |
| relief | 0.0086 | 0.075 | 0.995 | 1.148 |
| mean_slope_deg | 0.7424 | 0.109 | 0.993 | 1.081 |
| water_fraction | 0.0151 | 0.049 | 0.995 | 0.897 |
| spectral_beta | 0.8175 | 0.726 | 0.348 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.430 | 0.115 | 3.74 |
| hills | 0.364 | 0.129 | 2.83 |
| mountains | 0.366 | 0.131 | 2.79 |
| ridges | 0.325 | 0.102 | 3.18 |
| islands | 0.265 | 0.133 | 1.99 |
| mesas | 0.346 | 0.156 | 2.22 |

memorization: median NN-RMSE learned 0.0436 vs held-out 0.0432 (ratio 1.01); fraction of learned maps closer than held-out 1st percentile: 0.028

sampling: 742.6 ms/map

