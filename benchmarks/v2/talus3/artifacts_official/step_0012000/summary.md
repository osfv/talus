# Evaluation: checkpoints\talus-3.pt

step 12000, dataset `base64`/test, 1000 maps per half, guidance 2.0, 50 DDIM steps (quadratic spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.0747 | 0.0495 | 1.51 |
| rapsd_distance | 0.0937 | 0.0103 | 9.12 |
| height_w1 | 0.0085 | 0.0084 | 1.01 |
| slope_w1_deg | 0.4136 | 0.2503 | 1.65 |

traversability pass rate: learned 0.916, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1578, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0002 | 0.002 | 1.000 | 1.116 |
| relief | 0.0048 | 0.042 | 0.998 | 1.148 |
| mean_slope_deg | 0.3500 | 0.051 | 0.998 | 1.081 |
| water_fraction | 0.0129 | 0.042 | 0.996 | 0.897 |
| spectral_beta | 0.2889 | 0.256 | 0.966 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.134 | 0.115 | 1.17 |
| hills | 0.150 | 0.129 | 1.16 |
| mountains | 0.225 | 0.131 | 1.71 |
| ridges | 0.183 | 0.102 | 1.80 |
| islands | 0.161 | 0.133 | 1.21 |
| mesas | 0.222 | 0.156 | 1.43 |

memorization: median NN-RMSE learned 0.0428 vs held-out 0.0432 (ratio 0.99); fraction of learned maps closer than held-out 1st percentile: 0.011

sampling: 172.4 ms/map

