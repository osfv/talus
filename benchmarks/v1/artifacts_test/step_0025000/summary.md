# Evaluation: C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925-164759_diffusion64\checkpoints\step_0025000.pt

step 25000, dataset `base64`/test, 1000 maps per half, guidance 1.5, 200 DDIM steps (uniform spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1691 | 0.0495 | 3.42 |
| rapsd_distance | 0.2771 | 0.0103 | 26.98 |
| height_w1 | 0.0084 | 0.0084 | 1.01 |
| slope_w1_deg | 1.0393 | 0.2503 | 4.15 |

traversability pass rate: learned 0.911, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1595, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0016 | 0.014 | 1.000 | 1.116 |
| relief | 0.0096 | 0.085 | 0.994 | 1.148 |
| mean_slope_deg | 1.0441 | 0.153 | 0.988 | 1.081 |
| water_fraction | 0.0147 | 0.048 | 0.995 | 0.897 |
| spectral_beta | 0.9361 | 0.831 | 0.304 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.409 | 0.115 | 3.57 |
| hills | 0.375 | 0.129 | 2.91 |
| mountains | 0.391 | 0.131 | 2.98 |
| ridges | 0.353 | 0.102 | 3.46 |
| islands | 0.298 | 0.133 | 2.24 |
| mesas | 0.319 | 0.156 | 2.05 |

memorization: median NN-RMSE learned 0.0441 vs held-out 0.0432 (ratio 1.02); fraction of learned maps closer than held-out 1st percentile: 0.026

sampling: 746.3 ms/map

