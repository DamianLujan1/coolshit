# Backtest: walk-forward, seasons 2023, 2024, 2025

Every test week is predicted by a model trained only on earlier seasons and earlier weeks of the same season. Baselines use the closing line stored in the nflverse schedules file, which is never a model input.

## Season 2023

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.635 | 0.2262 | 0.6435 | 0.039 |
| Stage 2 (LightGBM) | 285 | 0.614 | 0.2371 | 0.6672 | 0.069 |
| Always home | 285 | 0.565 | 0.2460 | 0.6852 | 0.015 |
| Closing spread | 285 | 0.674 | 0.2173 | 0.6236 | 0.036 |
| Market (vig-free) | 285 | 0.670 | 0.2186 | 0.6270 | 0.043 |

## Season 2024

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.677 | 0.2113 | 0.6101 | 0.030 |
| Stage 2 (LightGBM) | 285 | 0.670 | 0.2104 | 0.6096 | 0.039 |
| Always home | 285 | 0.547 | 0.2478 | 0.6887 | 0.003 |
| Closing spread | 285 | 0.705 | 0.2039 | 0.5959 | 0.084 |
| Market (vig-free) | 285 | 0.709 | 0.2010 | 0.5892 | 0.061 |

## Season 2025

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 284 | 0.620 | 0.2206 | 0.6304 | 0.057 |
| Stage 2 (LightGBM) | 284 | 0.620 | 0.2282 | 0.6491 | 0.043 |
| Always home | 284 | 0.535 | 0.2490 | 0.6911 | 0.015 |
| Closing spread | 284 | 0.658 | 0.2116 | 0.6094 | 0.025 |
| Market (vig-free) | 284 | 0.658 | 0.2109 | 0.6070 | 0.047 |

## Combined

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 854 | 0.644 | 0.2194 | 0.6280 | 0.029 |
| Stage 2 (LightGBM) | 854 | 0.635 | 0.2252 | 0.6420 | 0.029 |
| Always home | 854 | 0.549 | 0.2476 | 0.6883 | 0.001 |
| Closing spread | 854 | 0.679 | 0.2109 | 0.6096 | 0.043 |
| Market (vig-free) | 854 | 0.679 | 0.2102 | 0.6077 | 0.025 |

## Calibration (combined, 10 bins)

### Stage 1 (logistic)

| Predicted home win prob | Games | Mean predicted | Observed home win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 1 | 0.098 | 0.000 |
| 0.1-0.2 | 22 | 0.166 | 0.273 |
| 0.2-0.3 | 55 | 0.258 | 0.309 |
| 0.3-0.4 | 123 | 0.357 | 0.358 |
| 0.4-0.5 | 132 | 0.453 | 0.447 |
| 0.5-0.6 | 164 | 0.550 | 0.506 |
| 0.6-0.7 | 164 | 0.652 | 0.689 |
| 0.7-0.8 | 123 | 0.747 | 0.724 |
| 0.8-0.9 | 62 | 0.842 | 0.806 |
| 0.9-1.0 | 8 | 0.916 | 1.000 |

### Stage 2 (LightGBM)

| Predicted home win prob | Games | Mean predicted | Observed home win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 7 | 0.186 | 0.286 |
| 0.2-0.3 | 66 | 0.264 | 0.364 |
| 0.3-0.4 | 138 | 0.352 | 0.377 |
| 0.4-0.5 | 146 | 0.450 | 0.438 |
| 0.5-0.6 | 134 | 0.545 | 0.515 |
| 0.6-0.7 | 143 | 0.649 | 0.629 |
| 0.7-0.8 | 131 | 0.745 | 0.725 |
| 0.8-0.9 | 88 | 0.845 | 0.818 |
| 0.9-1.0 | 1 | 0.909 | 1.000 |

### Market (vig-free)

| Predicted home win prob | Games | Mean predicted | Observed home win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 1 | 0.088 | 0.000 |
| 0.1-0.2 | 20 | 0.168 | 0.100 |
| 0.2-0.3 | 60 | 0.252 | 0.267 |
| 0.3-0.4 | 110 | 0.352 | 0.300 |
| 0.4-0.5 | 150 | 0.443 | 0.427 |
| 0.5-0.6 | 160 | 0.558 | 0.600 |
| 0.6-0.7 | 154 | 0.646 | 0.656 |
| 0.7-0.8 | 131 | 0.747 | 0.740 |
| 0.8-0.9 | 63 | 0.849 | 0.873 |
| 0.9-1.0 | 5 | 0.912 | 1.000 |

## Verdict

- Shipping **Stage 1 (logistic)**. Stage 2 did not beat stage 1 on both log loss and Brier, so the simpler model ships.
- **The model does not beat the closing line.** Market log loss 0.6077 vs model 0.6280; market accuracy 0.679 vs model 0.644. That is the expected outcome for a public-data model, and the public record will show it either way.
- Against the closing-spread favorite: model accuracy 0.644 vs 0.679; against always-home: 0.549.
- Calibration: expected calibration error 0.029 for the model vs 0.025 for the market. Lower is better; this is the number that matters most because picks are published as probabilities.

## Current season to date (not part of the holdout)

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Stage 1 (logistic) | 64 | 0.594 | 0.2315 | 0.6528 |
| Stage 2 (LightGBM) | 64 | 0.609 | 0.2445 | 0.6814 |
| Always home | 64 | 0.562 | 0.2463 | 0.6856 |
| Closing spread | 64 | 0.641 | 0.2281 | 0.6478 |
| Market (vig-free) | 64 | 0.641 | 0.2294 | 0.6509 |
