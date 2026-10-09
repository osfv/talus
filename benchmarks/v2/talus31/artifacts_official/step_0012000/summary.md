# Evaluation: checkpoints\talus-3.1.pt

step 12000, dataset `base64`/test, 1000 maps per half, guidance 2.0, 50 DDIM steps (quadratic spacing, eta 0.0)

## conditional

| metric | learned vs B | noise floor (A vs B) | ratio |
|---|---|---|---|
| metric_w1_mean | 0.0768 | 0.0495 | 1.55 |
| rapsd_distance | 0.0913 | 0.0103 | 8.89 |
| height_w1 | 0.0085 | 0.0084 | 1.02 |
| slope_w1_deg | 0.4242 | 0.2503 | 1.70 |

traversability pass rate: learned 0.911, procedural 0.917; diversity (mean pairwise RMSE): learned 0.1573, procedural 0.1662

| property | MAE | MAE / std | Pearson r | random-pair MAE / std |
|---|---|---|---|---|
| mean_elevation | 0.0002 | 0.002 | 1.000 | 1.116 |
| relief | 0.0050 | 0.044 | 0.998 | 1.148 |
| mean_slope_deg | 0.3116 | 0.046 | 0.999 | 1.081 |
| water_fraction | 0.0128 | 0.042 | 0.996 | 0.897 |
| spectral_beta | 0.2798 | 0.248 | 0.968 | 1.044 |

| archetype | metric W1 mean | floor | ratio |
|---|---|---|---|
| plains | 0.129 | 0.115 | 1.13 |
| hills | 0.155 | 0.129 | 1.20 |
| mountains | 0.230 | 0.131 | 1.75 |
| ridges | 0.197 | 0.102 | 1.93 |
| islands | 0.154 | 0.133 | 1.16 |
| mesas | 0.207 | 0.156 | 1.33 |

memorization: median NN-RMSE learned 0.0424 vs held-out 0.0432 (ratio 0.98); fraction of learned maps closer than held-out 1st percentile: 0.011

sampling: 178.7 ms/map

