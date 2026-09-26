# Evaluation: runs/20260926-192753_talus2_relative/checkpoints/step_0010000.pt

step 10000, dataset `base64`/test, 1000 maps per half, guidance 2.0, 50 DDIM steps (quadratic spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.0875 | 0.0495 | 1.77 |
| rapsd_distance | 0.1355 | 0.0103 | 13.19 |
| height_w1 | 0.0085 | 0.0084 | 1.02 |
| slope_w1_deg | 0.4437 | 0.2503 | 1.77 |

traversability pass rate: learned 0.895, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1570, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0004 | 0.003 | 1.000 | 1.116 |
| relief | 0.0051 | 0.045 | 0.998 | 1.148 |
| mean_slope_deg | 0.4479 | 0.066 | 0.997 | 1.081 |
| water_fraction | 0.0129 | 0.042 | 0.996 | 0.897 |
| spectral_beta | 0.3625 | 0.322 | 0.957 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.141 | 0.115 | 1.23 |
| hills | 0.180 | 0.129 | 1.40 |
| mountains | 0.253 | 0.131 | 1.93 |
| ridges | 0.220 | 0.102 | 2.16 |
| islands | 0.192 | 0.133 | 1.44 |
| mesas | 0.179 | 0.156 | 1.15 |

memorization: median NN-RMSE learned 0.0423 vs held-out 0.0432 (ratio 0.98); fraction of learned maps closer than held-out 1st percentile: 0.014

sampling: 179.3 ms/map

