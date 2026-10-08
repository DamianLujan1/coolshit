# Backtest: walk-forward, seasons 2023, 2024, 2025

Every test week is predicted by a model trained only on earlier seasons and earlier weeks of the same season. Baselines use the closing line stored in the nflverse schedules file, which is never a model input.

## Season 2023

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.635 | 0.2299 | 0.6521 | 0.064 |
| Stage 2 (LightGBM) | 285 | 0.611 | 0.2411 | 0.6789 | 0.081 |
| Always home | 285 | 0.565 | 0.2460 | 0.6852 | 0.015 |
| Closing spread | 285 | 0.674 | 0.2173 | 0.6236 | 0.036 |
| Market (vig-free) | 285 | 0.670 | 0.2186 | 0.6270 | 0.043 |

## Season 2024

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.688 | 0.2115 | 0.6109 | 0.047 |
| Stage 2 (LightGBM) | 285 | 0.681 | 0.2119 | 0.6144 | 0.076 |
| Always home | 285 | 0.547 | 0.2478 | 0.6887 | 0.003 |
| Closing spread | 285 | 0.705 | 0.2039 | 0.5959 | 0.084 |
| Market (vig-free) | 285 | 0.709 | 0.2010 | 0.5892 | 0.061 |

## Season 2025

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 284 | 0.616 | 0.2257 | 0.6424 | 0.057 |
| Stage 2 (LightGBM) | 284 | 0.620 | 0.2323 | 0.6572 | 0.086 |
| Always home | 284 | 0.535 | 0.2490 | 0.6911 | 0.015 |
| Closing spread | 284 | 0.658 | 0.2116 | 0.6094 | 0.025 |
| Market (vig-free) | 284 | 0.658 | 0.2109 | 0.6070 | 0.047 |

## Combined

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 854 | 0.646 | 0.2224 | 0.6351 | 0.031 |
| Stage 2 (LightGBM) | 854 | 0.637 | 0.2284 | 0.6502 | 0.039 |
| Always home | 854 | 0.549 | 0.2476 | 0.6883 | 0.001 |
| Closing spread | 854 | 0.679 | 0.2109 | 0.6096 | 0.043 |
| Market (vig-free) | 854 | 0.679 | 0.2102 | 0.6077 | 0.025 |

## Calibration (combined, 10 bins)

### Stage 1 (logistic)

| Predicted home win prob | Games | Mean predicted | Observed home win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 2 | 0.093 | 0.000 |
| 0.1-0.2 | 24 | 0.155 | 0.208 |
| 0.2-0.3 | 77 | 0.257 | 0.377 |
| 0.3-0.4 | 119 | 0.350 | 0.370 |
| 0.4-0.5 | 151 | 0.451 | 0.444 |
| 0.5-0.6 | 150 | 0.550 | 0.587 |
| 0.6-0.7 | 149 | 0.646 | 0.624 |
| 0.7-0.8 | 113 | 0.744 | 0.761 |
| 0.8-0.9 | 60 | 0.843 | 0.817 |
| 0.9-1.0 | 9 | 0.922 | 0.889 |

### Stage 2 (LightGBM)

| Predicted home win prob | Games | Mean predicted | Observed home win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 1 | 0.132 | 0.000 |
| 0.2-0.3 | 92 | 0.258 | 0.370 |
| 0.3-0.4 | 149 | 0.350 | 0.389 |
| 0.4-0.5 | 151 | 0.447 | 0.444 |
| 0.5-0.6 | 140 | 0.550 | 0.571 |
| 0.6-0.7 | 109 | 0.649 | 0.624 |
| 0.7-0.8 | 105 | 0.749 | 0.752 |
| 0.8-0.9 | 95 | 0.847 | 0.747 |
| 0.9-1.0 | 12 | 0.921 | 1.000 |

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
- **The model does not beat the closing line.** Market log loss 0.6077 vs model 0.6351; market accuracy 0.679 vs model 0.646. That is the expected outcome for a public-data model, and the public record will show it either way.
- Against the closing-spread favorite: model accuracy 0.646 vs 0.679; against always-home: 0.549.
- Calibration: expected calibration error 0.031 for the model vs 0.025 for the market. Lower is better; this is the number that matters most because picks are published as probabilities.

## Current season to date (not part of the holdout)

| Model | Games | Accuracy | Brier | Log loss |
|---|---:|---:|---:|---:|
| Stage 1 (logistic) | 64 | 0.609 | 0.2344 | 0.6598 |
| Stage 2 (LightGBM) | 64 | 0.594 | 0.2359 | 0.6634 |
| Always home | 64 | 0.562 | 0.2463 | 0.6856 |
| Closing spread | 64 | 0.641 | 0.2281 | 0.6478 |
| Market (vig-free) | 64 | 0.641 | 0.2294 | 0.6509 |
