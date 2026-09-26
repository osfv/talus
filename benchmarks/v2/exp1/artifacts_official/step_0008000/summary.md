# Evaluation: runs/20260926-181917_v2_exp1_nominsnr_cosine/checkpoints/step_0008000.pt

step 8000, dataset `base64`/test, 1000 maps per half, guidance 2.0, 50 DDIM steps (quadratic spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.1382 | 0.0495 | 2.79 |
| rapsd_distance | 0.3546 | 0.0103 | 34.52 |
| height_w1 | 0.0087 | 0.0084 | 1.03 |
| slope_w1_deg | 0.4174 | 0.2503 | 1.67 |

traversability pass rate: learned 0.895, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1583, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0010 | 0.009 | 1.000 | 1.116 |
| relief | 0.0057 | 0.050 | 0.998 | 1.148 |
| mean_slope_deg | 0.5994 | 0.088 | 0.994 | 1.081 |
| water_fraction | 0.0135 | 0.044 | 0.996 | 0.897 |
| spectral_beta | 0.6557 | 0.582 | 0.480 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.418 | 0.115 | 3.64 |
| hills | 0.224 | 0.129 | 1.74 |
| mountains | 0.250 | 0.131 | 1.90 |
| ridges | 0.202 | 0.102 | 1.98 |
| islands | 0.218 | 0.133 | 1.64 |
| mesas | 0.266 | 0.156 | 1.71 |

memorization: median NN-RMSE learned 0.0434 vs held-out 0.0432 (ratio 1.01); fraction of learned maps closer than held-out 1st percentile: 0.021

sampling: 177.4 ms/map

