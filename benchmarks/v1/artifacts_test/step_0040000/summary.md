# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-192102_diffusion64_cont\checkpoints\step_0040000.pt

step 40000, dataset `base64`/test, 1000 maps per half, guidance 1.5, 200 DDIM steps (uniform spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1492 | 0.0495 | 3.01 |
| rapsd_distance | 0.2528 | 0.0103 | 24.61 |
| height_w1 | 0.0082 | 0.0084 | 0.97 |
| slope_w1_deg | 0.8702 | 0.2503 | 3.48 |

traversability pass rate: learned 0.905, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1595, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0013 | 0.012 | 1.000 | 1.116 |
| relief | 0.0076 | 0.066 | 0.997 | 1.148 |
| mean_slope_deg | 0.7969 | 0.117 | 0.993 | 1.081 |
| water_fraction | 0.0135 | 0.044 | 0.996 | 0.897 |
| spectral_beta | 0.8208 | 0.729 | 0.383 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.449 | 0.115 | 3.91 |
| hills | 0.340 | 0.129 | 2.64 |
| mountains | 0.346 | 0.131 | 2.64 |
| ridges | 0.291 | 0.102 | 2.85 |
| islands | 0.284 | 0.133 | 2.13 |
| mesas | 0.270 | 0.156 | 1.73 |

memorization: median NN-RMSE learned 0.0422 vs held-out 0.0432 (ratio 0.98); fraction of learned maps closer than held-out 1st percentile: 0.031

sampling: 727.9 ms/map

