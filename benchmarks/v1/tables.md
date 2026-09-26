### Checkpoints (TEST, 1000 maps/half, 200-step DDIM uniform, guidance 1.5)

| step | metric W1 ratio | RAPSD ratio | height W1 ratio | slope W1 ratio | RAPSD bias (dec) | slope W1 excess (deg) | adherence nMAE | trav pass gen/ref | diversity gen/ref | NN ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 20000 | 3.19 | 23.0 | 1.13 | 2.94 | 0.236 | 0.49 | 0.240 | 0.911/0.917 | 0.1632/0.1662 | 1.005 |
| 25000 | 3.42 | 27.0 | 1.01 | 4.15 | 0.277 | 0.79 | 0.226 | 0.911/0.917 | 0.1595/0.1662 | 1.022 |
| 30000 | 3.00 | 24.0 | 1.09 | 2.86 | 0.246 | 0.47 | 0.189 | 0.907/0.917 | 0.1605/0.1662 | 1.015 |
| 35000 | 3.21 | 22.3 | 1.08 | 3.02 | 0.228 | 0.50 | 0.195 | 0.912/0.917 | 0.1601/0.1662 | 1.011 |
| 40000 | 3.01 | 24.6 | 0.97 | 3.48 | 0.253 | 0.62 | 0.194 | 0.905/0.917 | 0.1595/0.1662 | 0.978 |

Noise floor (A vs B, TEST): metric W1 0.0495, RAPSD 0.0103 decades, height W1 0.0084, slope W1 0.250 deg.


### Bootstrap 95% CIs (TEST)

| step | metric W1 ratio [95% CI] | RAPSD ratio [95% CI] | P(best metric W1) |
|---|---|---|---|
| 20000 | 2.43 [1.78, 3.14] | 6.2 [2.2, 20.2] | 0.00 |
| 25000 | 2.58 [1.86, 3.34] | 7.3 [2.6, 23.5] | 0.00 |
| 30000 | 2.27 [1.67, 2.95] | 6.3 [2.3, 20.4] | 0.66 |
| 35000 | 2.42 [1.76, 3.19] | 6.0 [2.2, 19.1] | 0.00 |
| 40000 | 2.29 [1.65, 2.97] | 6.5 [2.4, 20.8] | 0.34 |


### Artifacts and extrema (TEST means)

| step | checkerboard gen/ref | hf_energy gen/ref | peak density gen/ref | sink density gen/ref |
|---|---|---|---|---|
| 20000 | 0.0004/0.0005 | 0.0040/0.0062 | 16.18/11.93 | 11.18/8.40 |
| 25000 | 0.0004/0.0005 | 0.0037/0.0062 | 16.58/11.93 | 11.78/8.40 |
| 30000 | 0.0005/0.0005 | 0.0048/0.0062 | 18.06/11.93 | 12.81/8.40 |
| 35000 | 0.0004/0.0005 | 0.0046/0.0062 | 17.16/11.93 | 12.36/8.40 |
| 40000 | 0.0005/0.0005 | 0.0054/0.0062 | 18.56/11.93 | 12.50/8.40 |


### Conditioning adherence per property (TEST, nMAE = MAE / reference std; r = Pearson)

| step | mean_elevation nMAE / r | relief nMAE / r | mean_slope_deg nMAE / r | water_fraction nMAE / r | spectral_beta nMAE / r |
|---|---|---|---|---|---|
| 20000 | 0.033 / 1.000 | 0.094 / 0.994 | 0.140 / 0.985 | 0.052 / 0.995 | 0.880 / 0.269 |
| 25000 | 0.014 / 1.000 | 0.085 / 0.994 | 0.153 / 0.988 | 0.048 / 0.995 | 0.831 / 0.304 |
| 30000 | 0.016 / 1.000 | 0.075 / 0.995 | 0.116 / 0.991 | 0.051 / 0.994 | 0.689 / 0.363 |
| 35000 | 0.014 / 1.000 | 0.075 / 0.995 | 0.109 / 0.993 | 0.049 / 0.995 | 0.726 / 0.348 |
| 40000 | 0.012 / 1.000 | 0.066 / 0.997 | 0.117 / 0.993 | 0.044 / 0.996 | 0.729 / 0.383 |


### Per-archetype metric W1 ratio (TEST)

| step | plains | hills | mountains | ridges | islands | mesas |
|---|---|---|---|---|---|---|
| 20000 | 3.51 | 2.42 | 3.25 | 3.62 | 2.07 | 2.21 |
| 25000 | 3.57 | 2.91 | 2.98 | 3.46 | 2.24 | 2.05 |
| 30000 | 3.75 | 2.98 | 2.73 | 2.78 | 1.72 | 2.05 |
| 35000 | 3.74 | 2.83 | 2.79 | 3.18 | 1.99 | 2.22 |
| 40000 | 3.91 | 2.64 | 2.64 | 2.85 | 2.13 | 1.73 |


### Checkpoints on VAL (selection only, 1000 maps/half)

| step | metric W1 ratio | RAPSD ratio | height W1 ratio | slope W1 ratio | adherence nMAE |
|---|---|---|---|---|---|
| 20000 | 3.55 | 10.0 | 0.93 | 1.56 | 0.233 |
| 25000 | 3.67 | 12.0 | 0.89 | 2.85 | 0.221 |
| 30000 | 3.16 | 11.7 | 0.96 | 1.58 | 0.186 |
| 35000 | 3.46 | 10.5 | 0.98 | 1.67 | 0.190 |
| 40000 | 3.20 | 11.5 | 0.80 | 2.25 | 0.187 |


Selected checkpoint: **30000** (lowest VAL metric_w1_mean ratio (3.164); runner-up 40000 within 0.05 but higher mean headline ratio (4.429 vs 4.359)).


### Sampler grid (VAL, 500 maps/half, selected checkpoint; cost = isolated ms/map at batch 128)

| config | ms/map | maps/s | metric W1 ratio | RAPSD ratio | slope W1 ratio | adherence nMAE | hf_energy gen | sink density gen | frontier |
|---|---|---|---|---|---|---|---|---|---|
| s50_q_g1_e0 | 93.1 | 10.7 | 2.55 | 5.8 | 3.36 | 0.206 | 0.0039 | 12.08 | yes |
| s25_u_g1.5_e0 | 93.7 | 10.7 | 2.95 | 8.6 | 6.03 | 0.277 | 0.0035 | 11.64 |  |
| s25_q_g1.5_e0 | 93.7 | 10.7 | 2.45 | 6.3 | 3.53 | 0.214 | 0.0047 | 12.73 | yes |
| s50_u_g1.5_e0 | 182.5 | 5.5 | 2.44 | 6.7 | 3.66 | 0.226 | 0.0038 | 11.85 |  |
| s50_q_g1.5_e0 | 182.5 | 5.5 | 2.13 | 6.9 | 1.81 | 0.178 | 0.0049 | 13.43 |  |
| s50_q_g2_e0 | 182.5 | 5.5 | 1.95 | 8.2 | 0.76 | 0.169 | 0.0060 | 15.37 | yes |
| s100_u_g1_e0 | 190.0 | 5.3 | 2.53 | 5.7 | 3.31 | 0.219 | 0.0035 | 11.41 |  |
| s100_q_g1_e0 | 190.0 | 5.3 | 2.38 | 6.2 | 2.66 | 0.193 | 0.0040 | 12.29 |  |
| s100_u_g1.5_e0 | 360.9 | 2.8 | 2.14 | 6.4 | 2.10 | 0.193 | 0.0043 | 12.47 |  |
| s100_q_g1.5_e0 | 360.9 | 2.8 | 2.03 | 7.3 | 1.26 | 0.170 | 0.0050 | 13.86 |  |
| s100_u_g2_e0 | 360.9 | 2.8 | 1.95 | 7.3 | 1.17 | 0.183 | 0.0052 | 13.87 |  |
| s100_u_g3_e0 | 360.9 | 2.8 | 2.05 | 9.4 | 1.38 | 0.183 | 0.0073 | 17.40 |  |
| s100_q_g2_e0 | 360.9 | 2.8 | 1.90 | 8.5 | 0.46 | 0.165 | 0.0062 | 15.87 | yes |
| s100_q_g3_e0 | 360.9 | 2.8 | 2.41 | 10.9 | 2.26 | 0.181 | 0.0090 | 20.09 |  |
| s100_u_g1.5_e0.5 | 360.9 | 2.8 | 2.16 | 6.3 | 2.20 | 0.188 | 0.0039 | 11.48 |  |
| s100_u_g1.5_e1 | 360.9 | 2.8 | 2.63 | 5.8 | 3.74 | 0.195 | 0.0022 | 8.75 |  |
| s100_q_g1.5_e0.5 | 360.9 | 2.8 | 2.07 | 7.3 | 0.96 | 0.159 | 0.0046 | 13.69 |  |
| s100_q_g1.5_e1 | 360.9 | 2.8 | 2.17 | 6.4 | 1.75 | 0.155 | 0.0037 | 11.74 |  |
| s200_u_g1.5_e0 | 722.6 | 1.4 | 2.04 | 6.9 | 1.28 | 0.177 | 0.0047 | 13.17 |  |
| s200_q_g1.5_e0 | 722.6 | 1.4 | 2.00 | 7.6 | 0.99 | 0.167 | 0.0051 | 14.09 |  |

VAL reference: hf_energy 0.0060, sink density 8.40. Recommended: **s50_q_g2_e0**; best quality: s100_q_g2_e0.


### Sampler confirmation on TEST (1000 maps/half, selected checkpoint)

| config | metric W1 ratio | RAPSD ratio | height W1 ratio | slope W1 ratio | adherence nMAE | trav pass gen | hf_energy gen | sink density gen |
|---|---|---|---|---|---|---|---|---|
| s200_u_g1.5_e0 (default) | 3.00 | 24.0 | 1.09 | 2.86 | 0.189 | 0.907 | 0.0048 | 12.81 |
| s50_q_g1_e0 | 3.55 | 21.7 | 0.99 | 4.67 | 0.219 | 0.917 | 0.0043 | 11.60 |
| s25_q_g1.5_e0 | 3.66 | 24.4 | 1.17 | 5.09 | 0.229 | 0.909 | 0.0050 | 12.44 |
| s50_q_g2_e0 | 2.90 | 28.0 | 1.13 | 2.43 | 0.182 | 0.902 | 0.0061 | 14.64 |
| s100_q_g2_e0 | 2.82 | 28.9 | 1.12 | 1.93 | 0.177 | 0.904 | 0.0062 | 15.05 |


### Inference performance (isolated, RTX 5060, bf16)

| steps | guidance | batch | maps/s | s/map | latency/batch (s) | peak alloc MB | nvidia-smi MB |
|---|---|---|---|---|---|---|---|
| 25 | 1.0 | 1 | 1.21 | 0.8280 | 0.83 | 183 | 814 |
| 25 | 1.0 | 32 | 21.06 | 0.0475 | 1.52 | 452 | 1343 |
| 25 | 1.0 | 128 | 20.94 | 0.0478 | 6.11 | 1374 | 3176 |
| 25 | 1.0 | 256 | 21.43 | 0.0467 | 11.95 | 2595 | 4184 |
| 25 | 1.5 | 1 | 1.34 | 0.7435 | 0.74 | 185 | 814 |
| 25 | 1.5 | 32 | 11.04 | 0.0906 | 2.90 | 758 | 1929 |
| 25 | 1.5 | 128 | 10.67 | 0.0937 | 12.00 | 2592 | 4206 |
| 25 | 1.5 | 256 | OOM under cap |  |  |  |  |
| 50 | 1.0 | 1 | 0.68 | 1.4718 | 1.47 | 183 | 818 |
| 50 | 1.0 | 32 | 10.19 | 0.0981 | 3.14 | 452 | 1349 |
| 50 | 1.0 | 128 | 10.74 | 0.0931 | 11.92 | 1374 | 3136 |
| 50 | 1.0 | 256 | 10.78 | 0.0928 | 23.75 | 2595 | 4220 |
| 50 | 1.5 | 1 | 0.54 | 1.8390 | 1.84 | 185 | 823 |
| 50 | 1.5 | 32 | 5.36 | 0.1865 | 5.97 | 758 | 2007 |
| 50 | 1.5 | 128 | 5.48 | 0.1825 | 23.36 | 2592 | 4176 |
| 50 | 1.5 | 256 | OOM under cap |  |  |  |  |
| 100 | 1.0 | 1 | 0.36 | 2.7814 | 2.78 | 183 | 814 |
| 100 | 1.0 | 32 | 5.42 | 0.1844 | 5.90 | 452 | 1355 |
| 100 | 1.0 | 128 | 5.26 | 0.1900 | 24.31 | 1374 | 3145 |
| 100 | 1.0 | 256 | 5.31 | 0.1885 | 48.25 | 2595 | 4214 |
| 100 | 1.5 | 1 | 0.36 | 2.8104 | 2.81 | 185 | 839 |
| 100 | 1.5 | 32 | 2.81 | 0.3554 | 11.37 | 758 | 1806 |
| 100 | 1.5 | 128 | 2.77 | 0.3609 | 46.20 | 2592 | 4051 |
| 100 | 1.5 | 256 | OOM under cap |  |  |  |  |
| 200 | 1.0 | 1 | 0.18 | 5.6393 | 5.64 | 183 | 675 |
| 200 | 1.0 | 32 | 2.76 | 0.3621 | 11.59 | 452 | 1203 |
| 200 | 1.0 | 128 | 2.74 | 0.3647 | 46.68 | 1374 | 3236 |
| 200 | 1.0 | 256 | 2.76 | 0.3628 | 92.87 | 2595 | 4327 |
| 200 | 1.5 | 1 | 0.16 | 6.2152 | 6.22 | 185 | 936 |
| 200 | 1.5 | 32 | 1.45 | 0.6883 | 22.03 | 758 | 2049 |
| 200 | 1.5 | 128 | 1.38 | 0.7226 | 92.49 | 2592 | 4421 |
| 200 | 1.5 | 256 | OOM under cap |  |  |  |  |

GPU determinism: same seed + batch max |diff| 0.00e+00; batch 16 vs 1 max |diff| 3.46e-03 (4.147 m). Checkpoint load 1.0 s. CPU metrics: compute_metrics 2.1 ms/map, traversability 0.9 ms/map. Peak process RSS 0 MB.


### Training throughput (from run logs, batch 48)

| run | median it/s | steady median it/s | p10-p90 it/s | maps/s | slow intervals | wall h |
|---|---|---|---|---|---|---|
| main_0_30k | 4.27 | 4.29 | 3.37-4.45 | 206 | 0.110 | 2.22 |
| cont_30k_40k | 4.30 | 4.33 | 3.67-4.56 | 208 | 0.080 | 0.79 |


### Archetype failure modes (TEST, step 30000)

| archetype | metric W1 ratio | RAPSD ratio | slope W1 ratio | adherence nMAE | trav pass gen/ref | hf_energy gen/ref | sink gen/ref | worst metrics |
|---|---|---|---|---|---|---|---|---|
| plains | 3.75 | 26.8 | 5.76 | 0.380 | 1.00/1.00 | 27.21 | 4.55 | curvature_std, spectral_beta, peak_density |
| hills | 2.98 | 20.7 | 2.84 | 0.206 | 0.96/0.98 | 0.26 | 0.72 | curvature_std, spectral_beta, peak_density |
| mountains | 2.73 | 4.0 | 2.07 | 0.135 | 1.00/1.00 | 0.45 | 1.38 | spectral_beta, checkerboard, hf_energy |
| ridges | 2.78 | 9.8 | 8.74 | 0.126 | 0.97/0.95 | 0.48 | 1.28 | hf_energy, curvature_std, spectral_beta |
| islands | 1.72 | 4.2 | 0.89 | 0.185 | 0.57/0.59 | 0.46 | 1.02 | peak_density, spectral_beta, hf_energy |
| mesas | 2.05 | 3.8 | 0.86 | 0.108 | 0.99/0.99 | 0.60 | 1.37 | peak_density, spectral_beta, sink_density |


Largest contributors (leave-one-archetype-out deltas / error shares):

| failure | rank 1 | rank 2 | rank 3 |
|---|---|---|---|
| spectrum_error | plains | hills | islands |
| slope_error | ridges | mountains | hills |
| conditioning_error | plains | hills | islands |
| traversability_failures | islands | hills | plains |
| hf_energy_artifacts | plains | hills | mountains |
| sink_artifacts | plains | mesas | mountains |
| overall_metric_ratio | plains | hills | ridges |

| archetype | RAPSD delta (dec) | slope W1 delta (deg) | adherence error share | trav excess failure rate | hf_energy excess share | sink excess share |
|---|---|---|---|---|---|---|
| plains | 0.0122 | -0.095 | 0.317 | 0.000 | 1.000 | 0.661 |
| hills | -0.0172 | 0.050 | 0.186 | 0.018 | 0.000 | 0.000 |
| mountains | -0.0684 | 0.092 | 0.121 | 0.000 | 0.000 | 0.097 |
| ridges | -0.0795 | 0.234 | 0.104 | -0.019 | 0.000 | 0.068 |
| islands | -0.0230 | -0.158 | 0.182 | 0.022 | 0.000 | 0.001 |
| mesas | -0.0766 | -0.316 | 0.090 | -0.000 | 0.000 | 0.173 |


### Spectrum error by frequency band (mean log10 power learned - procedural, TEST)

| step | k1-4 (map-scale shape, >1 km) | k5-12 (valleys/ridges, 340-820 m) | k13-24 (hillslopes, 170-315 m) | k25-32 (finest, 128-165 m) |
|---|---|---|---|---|
| 20000 | 0.014 | -0.125 | -0.300 | 0.205 |
| 25000 | -0.003 | -0.224 | -0.303 | 0.271 |
| 30000 | 0.014 | -0.147 | -0.142 | 0.388 |
| 35000 | -0.010 | -0.156 | -0.199 | 0.305 |
| 40000 | 0.001 | -0.180 | -0.199 | 0.346 |
| floor A-B | 0.007 | -0.001 | -0.000 | 0.013 |

| archetype | k1-4 (map-scale shape, >1 km) | k5-12 (valleys/ridges, 340-820 m) | k13-24 (hillslopes, 170-315 m) | k25-32 (finest, 128-165 m) |
|---|---|---|---|---|
| plains | -0.059 | -0.590 | 0.883 | 3.472 |
| hills | 0.035 | -0.106 | -0.614 | 0.081 |
| mountains | -0.039 | -0.101 | -0.356 | -0.369 |
| ridges | 0.003 | -0.027 | -0.246 | -0.396 |
| islands | 0.032 | -0.074 | -0.287 | 0.141 |
| mesas | 0.112 | 0.062 | -0.063 | -0.381 |


### Adherence by requested-value bin (TEST, step 30000; bias in std units)

| property | bin | n | requested range | nMAE | bias |
|---|---|---|---|---|---|
| mean_elevation | lower | 334 | 0.067-0.266 | 0.030 | -0.027 |
| mean_elevation | middle | 333 | 0.267-0.343 | 0.008 | -0.001 |
| mean_elevation | upper | 333 | 0.343-0.603 | 0.011 | +0.006 |
| relief | lower | 334 | 0.023-0.173 | 0.050 | -0.019 |
| relief | middle | 333 | 0.173-0.285 | 0.084 | -0.013 |
| relief | upper | 333 | 0.285-0.510 | 0.090 | +0.018 |
| mean_slope_deg | lower | 334 | 1.343-9.931 | 0.087 | -0.052 |
| mean_slope_deg | middle | 333 | 9.932-15.025 | 0.128 | -0.087 |
| mean_slope_deg | upper | 333 | 15.085-32.984 | 0.134 | -0.067 |
| water_fraction | zero | 474 | 0.000-0.000 | 0.003 | +0.003 |
| water_fraction | low_nonzero | 263 | 0.000-0.115 | 0.063 | +0.021 |
| water_fraction | high_nonzero | 263 | 0.115-0.954 | 0.127 | +0.038 |
| spectral_beta | lower | 334 | 2.902-4.031 | 0.476 | +0.458 |
| spectral_beta | middle | 333 | 4.038-5.019 | 0.537 | +0.446 |
| spectral_beta | upper | 333 | 5.019-8.357 | 1.054 | -0.634 |


### Memorization (TEST, step 30000, 8 rotations/flips vs 45,000 training maps)

| set | p0.1 | p1 | p5 | p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| generated | 0.0065 | 0.0073 | 0.0097 | 0.0136 | 0.0438 | 0.0775 |
| held-out test | 0.0066 | 0.0080 | 0.0105 | 0.0142 | 0.0432 | 0.0770 |

nn_median_ratio 1.015; generated below held-out p1: 0.024; below held-out minimum: 0.001 (generated min 0.0057 vs held-out min 0.0063).

| archetype | gen median | held-out median | ratio | gen below held-out p1 |
|---|---|---|---|---|
| plains | 0.0125 | 0.0127 | 0.983 | 0.019 |
| hills | 0.0372 | 0.0366 | 1.015 | 0.023 |
| mountains | 0.0543 | 0.0549 | 0.987 | 0.018 |
| ridges | 0.0791 | 0.0801 | 0.987 | 0.006 |
| islands | 0.0418 | 0.0409 | 1.021 | 0.027 |
| mesas | 0.0439 | 0.0427 | 1.026 | 0.000 |


Quantization sensitivity (step 30000): float ratios {'metric_w1_mean': 3.0, 'rapsd_distance': 24.0, 'height_w1': 1.09, 'slope_w1_deg': 2.86} vs uint16 {'metric_w1_mean': 2.99, 'rapsd_distance': 24.0, 'height_w1': 1.09, 'slope_w1_deg': 2.86}; sink density gen float 12.81 / uint16 12.71 / ref 8.40.


### Conditioning stress tests (step 30000, sampler {'steps': 50, 'spacing': 'quadratic', 'guidance': 2.0, 'eta': 0.0})

Single property (only that property given, archetype unknown; 64 maps per case):


| property | level | requested | measured mean ± std | bias | nMAE | largest other-property deviation from natural (std) |
|---|---|---|---|---|---|---|
| mean_elevation | low | 0.113 | 0.104 ± 0.002 | -0.009 | 0.086 | relief -0.48 |
| mean_elevation | mid | 0.313 | 0.311 ± 0.000 | -0.003 | 0.024 | mean_slope_deg +0.61 |
| mean_elevation | high | 0.489 | 0.491 ± 0.002 | +0.003 | 0.026 | relief -0.72 |
| relief | low | 0.045 | 0.041 ± 0.004 | -0.004 | 0.039 | spectral_beta -1.88 |
| relief | mid | 0.220 | 0.207 ± 0.014 | -0.013 | 0.131 | mean_elevation +0.86 |
| relief | high | 0.411 | 0.446 ± 0.018 | +0.034 | 0.312 | mean_elevation +1.27 |
| mean_slope_deg | low | 2.763 | 2.101 ± 1.044 | -0.662 | 0.166 | spectral_beta -3.01 |
| mean_slope_deg | mid | 12.303 | 11.504 ± 0.973 | -0.799 | 0.160 | mean_elevation +0.69 |
| mean_slope_deg | high | 25.542 | 25.702 ± 0.942 | +0.161 | 0.103 | mean_elevation +0.34 |
| water_fraction | low | 0.000 | 0.007 ± 0.020 | +0.007 | 0.023 | mean_elevation +0.39 |
| water_fraction | mid | 0.108 | 0.016 ± 0.043 | -0.091 | 0.324 | mean_elevation +0.88 |
| water_fraction | high | 0.877 | 0.533 ± 0.337 | -0.344 | 1.187 | mean_elevation +0.72 |
| spectral_beta | low | 3.410 | 4.046 ± 0.319 | +0.636 | 0.576 | mean_elevation +0.55 |
| spectral_beta | mid | 4.569 | 5.007 ± 0.462 | +0.438 | 0.449 | mean_elevation +0.30 |
| spectral_beta | high | 7.118 | 5.900 ± 0.729 | -1.219 | 1.073 | relief +1.63 |


Pairs at extreme percentiles (32 maps per case):


| case | train support | nMAE a (alone) | nMAE b (alone) |
|---|---|---|---|
| mean_elevation=low|relief=low | 0 | 0.039 (0.086) | 0.167 (0.039) |
| mean_elevation=low|relief=high | 500 | 0.108 (0.086) | 0.166 (0.312) |
| mean_elevation=high|relief=low | 0 | 0.159 (0.026) | 0.208 (0.039) |
| mean_elevation=high|relief=high | 569 | 0.012 (0.026) | 0.138 (0.312) |
| mean_elevation=low|mean_slope_deg=low | 10 | 0.079 (0.086) | 0.120 (0.166) |
| mean_elevation=low|mean_slope_deg=high | 0 | 0.055 (0.086) | 0.654 (0.103) |
| mean_elevation=high|mean_slope_deg=low | 0 | 0.107 (0.026) | 0.123 (0.166) |
| mean_elevation=high|mean_slope_deg=high | 576 | 0.015 (0.026) | 0.204 (0.103) |
| mean_elevation=low|water_fraction=low | 0 | 0.023 (0.086) | 2.832 (0.023) |
| mean_elevation=low|water_fraction=high | 3425 | 0.079 (0.086) | 0.065 (1.187) |
| mean_elevation=high|water_fraction=low | 2180 | 0.023 (0.026) | 0.000 (0.023) |
| mean_elevation=high|water_fraction=high | 0 | 0.229 (0.026) | 2.896 (1.187) |
| mean_elevation=low|spectral_beta=low | 23 | 0.113 (0.086) | 0.529 (0.576) |
| mean_elevation=low|spectral_beta=high | 9 | 0.123 (0.086) | 1.073 (1.073) |
| mean_elevation=high|spectral_beta=low | 230 | 0.008 (0.026) | 0.418 (0.576) |
| mean_elevation=high|spectral_beta=high | 0 | 0.104 (0.026) | 1.567 (1.073) |
| relief=low|mean_slope_deg=low | 4850 | 0.076 (0.039) | 0.114 (0.166) |
| relief=low|mean_slope_deg=high | 0 | 0.696 (0.039) | 0.877 (0.103) |
| relief=high|mean_slope_deg=low | 0 | 2.093 (0.312) | 0.297 (0.166) |
| relief=high|mean_slope_deg=high | 581 | 0.360 (0.312) | 0.080 (0.103) |
| relief=low|water_fraction=low | 5125 | 0.029 (0.039) | 0.000 (0.023) |
| relief=low|water_fraction=high | 0 | 0.111 (0.039) | 0.336 (1.187) |
| relief=high|water_fraction=low | 1935 | 0.288 (0.312) | 0.018 (0.023) |
| relief=high|water_fraction=high | 622 | 0.324 (0.312) | 0.933 (1.187) |
| relief=low|spectral_beta=low | 7 | 0.030 (0.039) | 0.421 (0.576) |
| relief=low|spectral_beta=high | 829 | 0.068 (0.039) | 2.644 (1.073) |
| relief=high|spectral_beta=low | 179 | 0.248 (0.312) | 0.679 (0.576) |
| relief=high|spectral_beta=high | 0 | 0.349 (0.312) | 0.655 (1.073) |
| mean_slope_deg=low|water_fraction=low | 5052 | 0.125 (0.166) | 0.052 (0.023) |
| mean_slope_deg=low|water_fraction=high | 10 | 0.238 (0.166) | 0.287 (1.187) |
| mean_slope_deg=high|water_fraction=low | 1993 | 0.128 (0.103) | 0.007 (0.023) |
| mean_slope_deg=high|water_fraction=high | 0 | 0.288 (0.103) | 2.292 (1.187) |
| mean_slope_deg=low|spectral_beta=low | 13 | 0.133 (0.166) | 1.207 (0.576) |
| mean_slope_deg=low|spectral_beta=high | 868 | 0.132 (0.166) | 3.092 (1.073) |
| mean_slope_deg=high|spectral_beta=low | 260 | 0.197 (0.103) | 0.471 (0.576) |
| mean_slope_deg=high|spectral_beta=high | 0 | 0.283 (0.103) | 1.595 (1.073) |
| water_fraction=low|spectral_beta=low | 6813 | 0.003 (0.023) | 0.460 (0.576) |
| water_fraction=low|spectral_beta=high | 1054 | 0.086 (0.023) | 1.101 (1.073) |
| water_fraction=high|spectral_beta=low | 26 | 1.881 (1.187) | 1.336 (0.576) |
| water_fraction=high|spectral_beta=high | 12 | 0.706 (1.187) | 0.562 (1.073) |


Partial conditioning from real TEST maps (nMAE of known properties by number known):


| # known | mean_elevation | relief | mean_slope_deg | water_fraction | spectral_beta |
|---|---|---|---|---|---|
| 1 | 0.031 | 0.144 | 0.184 | 0.375 | 0.613 |
| 2 | 0.026 | 0.154 | 0.165 | 0.158 | 0.582 |
| 3 | 0.020 | 0.092 | 0.131 | 0.148 | 0.588 |
| 4 | 0.020 | 0.082 | 0.109 | 0.065 | 0.631 |
| 5 | 0.017 | 0.083 | 0.099 | 0.064 | 0.698 |


Within-archetype control gain (request p10 vs p90 of that archetype, label known; 1 = perfect):


| archetype | mean_elevation | relief | mean_slope_deg | water_fraction | spectral_beta |
|---|---|---|---|---|---|
| plains | 1.01 | 1.05 | 0.85 | 0.00 | -0.04 |
| hills | 1.03 | 0.96 | 0.92 | 0.32 | 0.79 |
| mountains | 1.02 | 1.13 | 1.13 | 0.47 | 1.05 |
| ridges | 1.05 | 1.36 | 1.03 | 0.00 | 0.82 |
| islands | 1.11 | 1.12 | 0.90 | 0.53 | 0.25 |
| mesas | 1.03 | 0.97 | 1.05 | 0.47 | 0.69 |


### Engineering

pytest: 90 tests, 0 failures, 0 errors, 54.0 s.


| command | exit | seconds |
|---|---|---|
| nullscape.exe gen-dataset --config configs/dataset/smoke.yaml --set name=eng_tiny --set n= | 0 | 6.41 |
| nullscape.exe viz --dataset C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\benchmarks\v1\engine | 0 | 4.82 |
| nullscape.exe export --dataset C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\data\base64 --ind | 0 | 3.61 |
| nullscape.exe sample --checkpoint C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925- | 0 | 7.12 |
| nullscape.exe evaluate --checkpoint C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\2026092 | 0 | 25.73 |
| nullscape.exe compare --checkpoint C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20260925 | 0 | 7.88 |
| nullscape.exe sampler-sweep --checkpoint C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\runs\20 | 0 | 11.4 |
| nullscape.exe train --config configs/train/smoke.yaml --set dataset=C:\Users\fearl\OneDriv | 0 | 26.53 |
| nullscape.exe artifacts --run C:\Users\fearl\OneDrive\Desktop\NULLSCAPE\benchmarks\v1\engi | 1 | 10.13 |


Dataset: manifest hash matches True; splits disjoint/covering True; regeneration bit-exact 100/100 (labels 100/100).

Exports: PNG16 exact True, R16 exact True (8192 bytes), NPY exact True, OBJ v/vn/f 4096/4096/7938 (expected 4096/4096/7938), OBJ height error 0.0005 m.


| input | unity_size | already 2^n+1 | upsampled although valid |
|---|---|---|---|
| 64 | 65 | False | False |
| 65 | 129 | True | True |
| 128 | 129 | False | False |
| 129 | 257 | True | True |
| 257 | 513 | True | True |
| 513 | 1025 | True | True |
| 1025 | 2049 | True | True |
