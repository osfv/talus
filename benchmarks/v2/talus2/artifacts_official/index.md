# Checkpoint artifacts: artifacts_official

Sampler: 50 DDIM steps (quadratic), eta 0.0, guidance 2.0; 1000 maps per half from `test`; fixed-seed set = 2 per archetype.

![progression](progression.png)

![metrics](progression_metrics.png)

| step | ratio metric_w1_mean | ratio rapsd_distance | ratio height_w1 | ratio slope_w1_deg | mean nMAE | NN ratio | trav pass (gen/ref) | dir |
|---|---|---|---|---|---|---|---|---|
| 10000 | 1.77 | 13.19 | 1.02 | 1.77 | 0.096 | 0.98 | 0.90/0.92 | [step_0010000](step_0010000/summary.md) |
