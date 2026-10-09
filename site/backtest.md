# Backtest: 2022 selection and frozen 2023–2025 audit

The fixed logistic and boosting candidates are compared using 2022 only. Boosting is selected only if both its Brier score and log loss improve. The selected stage is then frozen before the 2023–2025 walk-forward audit. Each weekly fit uses only earlier-week games available by that week's prediction cutoff.

Exclude ties and unfinished games from binary training and scoring; published tied picks are pushes. Probabilities are conditional on a decisive result. A 0.5 probability picks home.

Always-home accuracy always picks home; its probability is the fixed pre-2022 decisive home-win rate. Closing-spread favorite uses sign (pick'em chooses home) and a fixed normal margin sigma of 13.5 for probability scores. No-vig moneylines use proportional normalization. Odds are benchmarks only, never features or selection inputs.

## 2022 model selection (not part of audit)

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 282 | 0.649 | 0.2229 | 0.6374 | 0.042 |
| Stage 2 (LightGBM) | 282 | 0.624 | 0.2323 | 0.6596 | 0.092 |
| Always home / historical home rate | 282 | 0.567 | 0.2459 | 0.6849 | 0.020 |
| Closing-spread favorite / normal probability proxy | 282 | 0.663 | 0.2101 | 0.6071 | 0.044 |
| Market (vig-free moneylines) | 282 | 0.656 | 0.2093 | 0.6050 | 0.048 |

## Audit season 2023

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.618 | 0.2297 | 0.6507 | 0.044 |
| Stage 2 (LightGBM) | 285 | 0.600 | 0.2406 | 0.6750 | 0.086 |
| Always home / historical home rate | 285 | 0.565 | 0.2461 | 0.6853 | 0.018 |
| Closing-spread favorite / normal probability proxy | 285 | 0.674 | 0.2173 | 0.6236 | 0.036 |
| Market (vig-free moneylines) | 285 | 0.670 | 0.2186 | 0.6270 | 0.043 |

## Audit season 2024

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 285 | 0.670 | 0.2158 | 0.6223 | 0.052 |
| Stage 2 (LightGBM) | 285 | 0.674 | 0.2162 | 0.6240 | 0.048 |
| Always home / historical home rate | 285 | 0.547 | 0.2478 | 0.6887 | 0.000 |
| Closing-spread favorite / normal probability proxy | 285 | 0.705 | 0.2039 | 0.5959 | 0.084 |
| Market (vig-free moneylines) | 285 | 0.709 | 0.2010 | 0.5892 | 0.061 |

## Audit season 2025

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 284 | 0.609 | 0.2236 | 0.6363 | 0.068 |
| Stage 2 (LightGBM) | 284 | 0.630 | 0.2315 | 0.6567 | 0.047 |
| Always home / historical home rate | 284 | 0.535 | 0.2489 | 0.6910 | 0.012 |
| Closing-spread favorite / normal probability proxy | 284 | 0.658 | 0.2116 | 0.6094 | 0.025 |
| Market (vig-free moneylines) | 284 | 0.658 | 0.2109 | 0.6070 | 0.047 |

## Audit combined (each baseline's available coverage)

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 854 | 0.632 | 0.2230 | 0.6364 | 0.021 |
| Stage 2 (LightGBM) | 854 | 0.635 | 0.2294 | 0.6519 | 0.050 |
| Always home / historical home rate | 854 | 0.549 | 0.2476 | 0.6883 | 0.002 |
| Closing-spread favorite / normal probability proxy | 854 | 0.679 | 0.2109 | 0.6096 | 0.043 |
| Market (vig-free moneylines) | 854 | 0.679 | 0.2102 | 0.6077 | 0.025 |

## Audit comparison on identical games: market moneylines

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 854 | 0.632 | 0.2230 | 0.6364 | 0.021 |
| Stage 2 (LightGBM) | 854 | 0.635 | 0.2294 | 0.6519 | 0.050 |
| Always home / historical home rate | 854 | 0.549 | 0.2476 | 0.6883 | 0.002 |
| Market (vig-free moneylines) | 854 | 0.679 | 0.2102 | 0.6077 | 0.025 |

## Audit comparison on identical games: closing-spread favorite

| Model | Games | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| Stage 1 (logistic) | 854 | 0.632 | 0.2230 | 0.6364 | 0.021 |
| Stage 2 (LightGBM) | 854 | 0.635 | 0.2294 | 0.6519 | 0.050 |
| Always home / historical home rate | 854 | 0.549 | 0.2476 | 0.6883 | 0.002 |
| Closing-spread favorite / normal probability proxy | 854 | 0.679 | 0.2109 | 0.6096 | 0.043 |

Audit coverage: 854 decisive model games; 854 with valid two-sided moneylines; 854 with valid spreads. Model scoring retains games without odds.

## Sample accounting

| Season | Completed | Ties excluded | Decisive | Modeled | Other exclusions |
|---|---:|---:|---:|---:|---:|
| 2022 | 284 | 2 | 282 | 282 | 0 |
| 2023 | 285 | 0 | 285 | 285 | 0 |
| 2024 | 285 | 0 | 285 | 285 | 0 |
| 2025 | 285 | 1 | 284 | 284 | 0 |

## Verdict

- Selected **Stage 1 (logistic)** using 2022 only. Boosting did not improve both selection metrics, or was not evaluated; logistic remains selected.
- **The market beats or ties the model on accuracy, Brier and log loss.** On 854 identical audit games: model/market accuracy 0.632/0.679, Brier 0.2230/0.2102, log loss 0.6364/0.6077.
- Matched calibration ECE: model 0.021, market 0.025. Ten-bin calibration is descriptive, sensitive to sample size, and is not a model-selection criterion.
- On 854 spread-covered games, model accuracy is 0.632 versus favorite 0.679; always-home accuracy across all modeled audit games is 0.549.
- Audit results do not change the selected stage. Both models are disclosed even when the selected model loses to the market.

## Historical reconstruction limits

- Historical source files were rebuilt from the latest nflverse release; original Tuesday data vintages and correction arrival times are unavailable.
- Historical prediction cutoff is Tuesday 12:00 America/New_York before the week's first game. Completed history requires the conservative kickoff-plus-12-hours availability proxy; this is not a verified publication timestamp.
- QB inputs use only timestamped pre-cutoff observations when supplied, otherwise a strictly lagged completed-game QB prior. Current schedule actual starters cannot establish historical Tuesday availability.
- The candidates and 2022 selection rule are held fixed for this rebuilt audit. This retrospective repair is not an untouched or prospective validation; 2023–2025 results were examined in the earlier flawed build.
- Closing market lines can contain information unavailable on Tuesday. They are strong later-information benchmarks, not contemporaneous tradable prices.

## Calibration (10 equal-width probability bins)

### Selection 2022: Stage 1 (logistic)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 5 | 0.172 | 0.400 |
| 0.2-0.3 | 15 | 0.258 | 0.400 |
| 0.3-0.4 | 39 | 0.359 | 0.359 |
| 0.4-0.5 | 42 | 0.449 | 0.405 |
| 0.5-0.6 | 60 | 0.546 | 0.567 |
| 0.6-0.7 | 67 | 0.648 | 0.701 |
| 0.7-0.8 | 37 | 0.748 | 0.730 |
| 0.8-0.9 | 16 | 0.829 | 0.750 |
| 0.9-1.0 | 1 | 0.908 | 1.000 |

### Selection 2022: Stage 2 (LightGBM)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 7 | 0.182 | 0.571 |
| 0.2-0.3 | 25 | 0.256 | 0.480 |
| 0.3-0.4 | 45 | 0.355 | 0.333 |
| 0.4-0.5 | 45 | 0.453 | 0.489 |
| 0.5-0.6 | 41 | 0.551 | 0.634 |
| 0.6-0.7 | 54 | 0.652 | 0.574 |
| 0.7-0.8 | 46 | 0.752 | 0.826 |
| 0.8-0.9 | 19 | 0.842 | 0.632 |
| 0.9-1.0 | 0 | n/a | n/a |

### Selection 2022: Always home / historical home rate

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a |
| 0.2-0.3 | 0 | n/a | n/a |
| 0.3-0.4 | 0 | n/a | n/a |
| 0.4-0.5 | 0 | n/a | n/a |
| 0.5-0.6 | 282 | 0.547 | 0.567 |
| 0.6-0.7 | 0 | n/a | n/a |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### Selection 2022: Closing-spread favorite / normal probability proxy

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 4 | 0.152 | 0.250 |
| 0.2-0.3 | 14 | 0.259 | 0.286 |
| 0.3-0.4 | 35 | 0.359 | 0.343 |
| 0.4-0.5 | 50 | 0.439 | 0.420 |
| 0.5-0.6 | 69 | 0.568 | 0.536 |
| 0.6-0.7 | 64 | 0.647 | 0.688 |
| 0.7-0.8 | 36 | 0.757 | 0.861 |
| 0.8-0.9 | 10 | 0.860 | 1.000 |
| 0.9-1.0 | 0 | n/a | n/a |

### Selection 2022: Market (vig-free moneylines)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 5 | 0.158 | 0.400 |
| 0.2-0.3 | 18 | 0.252 | 0.222 |
| 0.3-0.4 | 40 | 0.356 | 0.400 |
| 0.4-0.5 | 38 | 0.454 | 0.421 |
| 0.5-0.6 | 58 | 0.558 | 0.552 |
| 0.6-0.7 | 59 | 0.646 | 0.593 |
| 0.7-0.8 | 49 | 0.752 | 0.837 |
| 0.8-0.9 | 14 | 0.846 | 0.929 |
| 0.9-1.0 | 1 | 0.905 | 1.000 |

### Audit 2023: Stage 1 (logistic)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 9 | 0.170 | 0.333 |
| 0.2-0.3 | 15 | 0.256 | 0.400 |
| 0.3-0.4 | 30 | 0.357 | 0.367 |
| 0.4-0.5 | 49 | 0.452 | 0.490 |
| 0.5-0.6 | 63 | 0.549 | 0.524 |
| 0.6-0.7 | 56 | 0.651 | 0.679 |
| 0.7-0.8 | 42 | 0.741 | 0.667 |
| 0.8-0.9 | 20 | 0.836 | 0.850 |
| 0.9-1.0 | 1 | 0.901 | 1.000 |

### Audit 2023: Stage 2 (LightGBM)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 2 | 0.175 | 0.000 |
| 0.2-0.3 | 21 | 0.257 | 0.381 |
| 0.3-0.4 | 40 | 0.358 | 0.575 |
| 0.4-0.5 | 51 | 0.457 | 0.412 |
| 0.5-0.6 | 42 | 0.547 | 0.595 |
| 0.6-0.7 | 63 | 0.652 | 0.587 |
| 0.7-0.8 | 41 | 0.747 | 0.659 |
| 0.8-0.9 | 25 | 0.837 | 0.800 |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2023: Always home / historical home rate

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a |
| 0.2-0.3 | 0 | n/a | n/a |
| 0.3-0.4 | 0 | n/a | n/a |
| 0.4-0.5 | 0 | n/a | n/a |
| 0.5-0.6 | 285 | 0.547 | 0.565 |
| 0.6-0.7 | 0 | n/a | n/a |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2023: Closing-spread favorite / normal probability proxy

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 5 | 0.172 | 0.200 |
| 0.2-0.3 | 10 | 0.250 | 0.200 |
| 0.3-0.4 | 31 | 0.361 | 0.355 |
| 0.4-0.5 | 63 | 0.431 | 0.397 |
| 0.5-0.6 | 60 | 0.571 | 0.650 |
| 0.6-0.7 | 71 | 0.639 | 0.662 |
| 0.7-0.8 | 30 | 0.742 | 0.767 |
| 0.8-0.9 | 14 | 0.846 | 0.857 |
| 0.9-1.0 | 1 | 0.903 | 1.000 |

### Audit 2023: Market (vig-free moneylines)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 9 | 0.167 | 0.111 |
| 0.2-0.3 | 12 | 0.245 | 0.417 |
| 0.3-0.4 | 31 | 0.357 | 0.290 |
| 0.4-0.5 | 56 | 0.441 | 0.429 |
| 0.5-0.6 | 58 | 0.561 | 0.638 |
| 0.6-0.7 | 57 | 0.645 | 0.649 |
| 0.7-0.8 | 43 | 0.752 | 0.767 |
| 0.8-0.9 | 18 | 0.855 | 0.778 |
| 0.9-1.0 | 1 | 0.920 | 1.000 |

### Audit 2024: Stage 1 (logistic)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 5 | 0.163 | 0.400 |
| 0.2-0.3 | 15 | 0.256 | 0.133 |
| 0.3-0.4 | 37 | 0.352 | 0.324 |
| 0.4-0.5 | 58 | 0.443 | 0.414 |
| 0.5-0.6 | 48 | 0.553 | 0.521 |
| 0.6-0.7 | 59 | 0.647 | 0.729 |
| 0.7-0.8 | 37 | 0.749 | 0.757 |
| 0.8-0.9 | 23 | 0.830 | 0.739 |
| 0.9-1.0 | 3 | 0.921 | 1.000 |

### Audit 2024: Stage 2 (LightGBM)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 1 | 0.195 | 1.000 |
| 0.2-0.3 | 19 | 0.262 | 0.263 |
| 0.3-0.4 | 41 | 0.350 | 0.366 |
| 0.4-0.5 | 59 | 0.447 | 0.356 |
| 0.5-0.6 | 46 | 0.553 | 0.587 |
| 0.6-0.7 | 47 | 0.654 | 0.617 |
| 0.7-0.8 | 46 | 0.755 | 0.804 |
| 0.8-0.9 | 26 | 0.852 | 0.808 |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2024: Always home / historical home rate

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a |
| 0.2-0.3 | 0 | n/a | n/a |
| 0.3-0.4 | 0 | n/a | n/a |
| 0.4-0.5 | 0 | n/a | n/a |
| 0.5-0.6 | 285 | 0.547 | 0.547 |
| 0.6-0.7 | 0 | n/a | n/a |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2024: Closing-spread favorite / normal probability proxy

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 1 | 0.111 | 0.000 |
| 0.2-0.3 | 9 | 0.276 | 0.222 |
| 0.3-0.4 | 52 | 0.353 | 0.212 |
| 0.4-0.5 | 53 | 0.434 | 0.415 |
| 0.5-0.6 | 59 | 0.573 | 0.559 |
| 0.6-0.7 | 70 | 0.654 | 0.814 |
| 0.7-0.8 | 32 | 0.741 | 0.688 |
| 0.8-0.9 | 8 | 0.844 | 1.000 |
| 0.9-1.0 | 1 | 0.926 | 1.000 |

### Audit 2024: Market (vig-free moneylines)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 1 | 0.088 | 0.000 |
| 0.1-0.2 | 1 | 0.160 | 0.000 |
| 0.2-0.3 | 23 | 0.261 | 0.217 |
| 0.3-0.4 | 43 | 0.352 | 0.209 |
| 0.4-0.5 | 46 | 0.446 | 0.435 |
| 0.5-0.6 | 55 | 0.560 | 0.582 |
| 0.6-0.7 | 49 | 0.649 | 0.776 |
| 0.7-0.8 | 47 | 0.742 | 0.723 |
| 0.8-0.9 | 19 | 0.833 | 0.895 |
| 0.9-1.0 | 1 | 0.926 | 1.000 |

### Audit 2025: Stage 1 (logistic)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 6 | 0.179 | 0.167 |
| 0.2-0.3 | 24 | 0.268 | 0.333 |
| 0.3-0.4 | 29 | 0.356 | 0.379 |
| 0.4-0.5 | 58 | 0.450 | 0.483 |
| 0.5-0.6 | 54 | 0.545 | 0.481 |
| 0.6-0.7 | 49 | 0.649 | 0.510 |
| 0.7-0.8 | 43 | 0.749 | 0.837 |
| 0.8-0.9 | 19 | 0.840 | 0.789 |
| 0.9-1.0 | 2 | 0.915 | 1.000 |

### Audit 2025: Stage 2 (LightGBM)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 5 | 0.185 | 0.600 |
| 0.2-0.3 | 17 | 0.267 | 0.353 |
| 0.3-0.4 | 51 | 0.346 | 0.373 |
| 0.4-0.5 | 48 | 0.442 | 0.396 |
| 0.5-0.6 | 52 | 0.545 | 0.519 |
| 0.6-0.7 | 43 | 0.660 | 0.605 |
| 0.7-0.8 | 37 | 0.742 | 0.757 |
| 0.8-0.9 | 31 | 0.836 | 0.774 |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2025: Always home / historical home rate

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a |
| 0.2-0.3 | 0 | n/a | n/a |
| 0.3-0.4 | 0 | n/a | n/a |
| 0.4-0.5 | 0 | n/a | n/a |
| 0.5-0.6 | 284 | 0.547 | 0.535 |
| 0.6-0.7 | 0 | n/a | n/a |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2025: Closing-spread favorite / normal probability proxy

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 3 | 0.171 | 0.000 |
| 0.2-0.3 | 17 | 0.260 | 0.235 |
| 0.3-0.4 | 49 | 0.357 | 0.327 |
| 0.4-0.5 | 50 | 0.431 | 0.440 |
| 0.5-0.6 | 53 | 0.566 | 0.566 |
| 0.6-0.7 | 67 | 0.647 | 0.642 |
| 0.7-0.8 | 22 | 0.736 | 0.682 |
| 0.8-0.9 | 23 | 0.840 | 0.957 |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit 2025: Market (vig-free moneylines)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 10 | 0.169 | 0.100 |
| 0.2-0.3 | 25 | 0.247 | 0.240 |
| 0.3-0.4 | 36 | 0.346 | 0.417 |
| 0.4-0.5 | 48 | 0.441 | 0.417 |
| 0.5-0.6 | 47 | 0.554 | 0.574 |
| 0.6-0.7 | 48 | 0.645 | 0.542 |
| 0.7-0.8 | 41 | 0.749 | 0.732 |
| 0.8-0.9 | 26 | 0.857 | 0.923 |
| 0.9-1.0 | 3 | 0.905 | 1.000 |

### Audit combined: Stage 1 (logistic)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 20 | 0.171 | 0.300 |
| 0.2-0.3 | 54 | 0.261 | 0.296 |
| 0.3-0.4 | 96 | 0.355 | 0.354 |
| 0.4-0.5 | 165 | 0.448 | 0.461 |
| 0.5-0.6 | 165 | 0.549 | 0.509 |
| 0.6-0.7 | 164 | 0.649 | 0.646 |
| 0.7-0.8 | 122 | 0.746 | 0.754 |
| 0.8-0.9 | 62 | 0.835 | 0.790 |
| 0.9-1.0 | 6 | 0.915 | 1.000 |

### Audit combined: Stage 2 (LightGBM)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 8 | 0.184 | 0.500 |
| 0.2-0.3 | 57 | 0.262 | 0.333 |
| 0.3-0.4 | 132 | 0.351 | 0.432 |
| 0.4-0.5 | 158 | 0.449 | 0.386 |
| 0.5-0.6 | 140 | 0.549 | 0.564 |
| 0.6-0.7 | 153 | 0.655 | 0.601 |
| 0.7-0.8 | 124 | 0.748 | 0.742 |
| 0.8-0.9 | 82 | 0.841 | 0.793 |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit combined: Always home / historical home rate

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 0 | n/a | n/a |
| 0.2-0.3 | 0 | n/a | n/a |
| 0.3-0.4 | 0 | n/a | n/a |
| 0.4-0.5 | 0 | n/a | n/a |
| 0.5-0.6 | 854 | 0.547 | 0.549 |
| 0.6-0.7 | 0 | n/a | n/a |
| 0.7-0.8 | 0 | n/a | n/a |
| 0.8-0.9 | 0 | n/a | n/a |
| 0.9-1.0 | 0 | n/a | n/a |

### Audit combined: Closing-spread favorite / normal probability proxy

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
|---|---:|---:|---:|
| 0.0-0.1 | 0 | n/a | n/a |
| 0.1-0.2 | 9 | 0.165 | 0.111 |
| 0.2-0.3 | 36 | 0.261 | 0.222 |
| 0.3-0.4 | 132 | 0.356 | 0.288 |
| 0.4-0.5 | 166 | 0.432 | 0.416 |
| 0.5-0.6 | 172 | 0.570 | 0.593 |
| 0.6-0.7 | 208 | 0.647 | 0.707 |
| 0.7-0.8 | 84 | 0.740 | 0.714 |
| 0.8-0.9 | 45 | 0.843 | 0.933 |
| 0.9-1.0 | 2 | 0.914 | 1.000 |

### Audit combined: Market (vig-free moneylines)

| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |
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
