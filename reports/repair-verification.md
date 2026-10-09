# Local repair verification — 2026-10-09

Status: verified local repair, unpublished. Branch `repair/causal-qb-weekly-safety`, based on `5caa7a732942721e3822a035b727d52902c11ab1`. No push or deployment was performed. No SPE9 or OffsetIQ files were changed.

## Checkout and preservation

The delegated task folder was empty. Searches in the Mac's user/project folders found no existing `coolshit`/`gridiron` checkout, and the app returned no saved repository project or attached worktree. A public Git clone was created at `/Users/damianlujan/Documents/Codex/2026-10-08/task/coolshit`, inspected clean at the stated main SHA, then switched to the repair branch. There were no uncommitted changes in this checkout to preserve. No repository `.agents`, `AGENTS.md`, or local memory-summary file was found in the inspected locations.

The system Git shim failed because the Xcode license was not accepted. `/Library/Developer/CommandLineTools/usr/bin/git` worked; no system license or configuration was changed. Initial sandbox DNS failures were resolved through approved network execution for the clone and public dependency/data downloads.

All 15 original Week 5 picks remain unchanged. `data/picks/picks.csv` and `site/picks.csv` SHA256: `a5039d5c2070268e663537a1daad8a3c0a7c81a2caf4a3e8c44fd95b808c39a5`. Original graded data, record JSON, and recap also remain unchanged. See `preserved-history.json`; Git diff on those paths is empty. The locally rendered HTML explains that old version 0.1.0 picks were not repaired retrospectively.

## Executed verification

Working directory for all commands below: the checkout above. Every command listed as passed actually executed.

| Command | Result |
| --- | --- |
| `UV_CACHE_DIR=/tmp/gridiron-uv-cache /Users/damianlujan/.local/bin/uv sync --locked --extra dev --offline` | Pinned environment synchronized; project 0.1.1, Python 3.14.6; logs in `verification/locked-install.txt` |
| `.venv/bin/python -m pytest` | **109 passed in 2.58s**; log `verification/pytest.txt` |
| `.venv/bin/python scripts/verify_cached_causality.py --season 2023 --week 5 --output reports/causality-check.json` | **6 real-data checks passed**, 14 target games each; exact QB/provenance equality, maximum feature difference `5.55e-17` |
| `.venv/bin/python -m compileall -q gridiron` | Exit 0 |
| `.venv/bin/python -m gridiron validate` | Offline four-dataset validation passed |
| `.venv/bin/python -m gridiron backtest --offline` | Both models completed 2022 selection, then frozen 2023–2025 audit; raw predictions and source/data hashes saved |
| `.venv/bin/python -m gridiron --now 2026-10-07T21:00:00+00:00 weekly --offline --dry-run --output-dir /tmp/gridiron-rehearsal-week5 --season 2026 --week 5` | Complete isolated workflow, 3,082 training games, 15 candidate picks, **0 appended**; existing bytes identical |
| `.venv/bin/python -m gridiron weekly --dry-run --output-dir /tmp/gridiron-rehearsal-week6 --season 2026 --week 6` | Complete **online** isolated workflow, fresh fetch/validate/retrain, **14 simulated new picks**, 14 QB evidence records, grade/record/recap/site; no Git or deployment subprocess |
| `/Library/Developer/CommandLineTools/usr/bin/git diff --check` | Passed before commit |

The first 14 legacy tests passed before the new regressions, demonstrating that they did not catch the identity leak. The expanded suite covers present/missing QB values, changed identities/dropbacks, removed target/future PBP, seed history, postponed/in-progress games, evidence capture times, missing odds, tie treatment, stage selection isolation, dry-run writes, and publication command ordering. Remote publication tests use mocks, not a live deployment.

Both rehearsal folders contain `DRY_RUN.json` with `publishable: false`. Week 6's copied log has 29 rows (15 originals + 14 simulations) and must not be published. The canonical log still has 15 rows and no invented provenance entries.

A direct `site.build_site` run rendered all three local pages using the preserved original picks, graded CSV and record plus the rebuilt report. Assertions verified 15 pick table rows, nflverse attribution, entertainment footer, no unresolved template expressions, unchanged canonical record bytes and byte-identical site CSV. This rendered local output has not been published.

The real-data checks mutate identities, suppress dominant-passer dropbacks and truncate target/future PBP, each with present and missing schedule starter values. Normalized snapshot hashes match; the `1e-12` numeric tolerance only accommodates floating-point reduction order. See `scripts/verify_cached_causality.py` and `causality-check.json`.

The final report's recorded source, cached data and raw prediction SHA256 values were recomputed and verified after the run.

## Data coverage

2014–2026 cache: **3,567 scheduled games, 591,160 plays, 6,718 team-stat rows, 221,939 player-stat rows**. The modeling period is 2015–2026; 2014 is a seed only. There are 40 cached parquet datasets. **261** upstream player-stat rows have no player ID; they are retained and reported, never used as QB identity evidence. Data provenance is in `data-manifest.json`. The initial PBP cache was populated before the manifest repair; its precise retrieval times are intentionally marked unknown rather than backfilled from file modification time.

FTN and participation datasets are excluded. Team/player statistics are cache/validation inputs, not additional model features. Market odds are only benchmarks. Missing kickoff information now fails safely.

## Rebuilt results

2022 selection uses 282 decisive games (284 completed, 2 ties excluded). Logistic Brier **0.2229**, log loss **0.6374** versus LightGBM **0.2323**, **0.6596**. Logistic is frozen as the shipping stage before the audit.

| Audit model | Games | Accuracy | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic (selected) | 854 | 63.23% | 0.2230 | 0.6364 |
| LightGBM | 854 | 63.47% | 0.2294 | 0.6519 |
| Always home | 854 | 54.92% | 0.2476 | 0.6883 |
| Closing favorite | 854 | 67.92% | 0.2109 | 0.6096 |
| No-vig market | 854 | 67.92% | 0.2102 | 0.6077 |

**The market wins** on accuracy, Brier and log loss. All 854 audit games have both market and spread benchmarks, so the comparisons cover identical games. 2023: 285 games; 2024: 285; 2025: 284 decisive games after excluding one tie. There are no other feature/sample exclusions. The report includes regular season and postseason, every season and combined results, and ten bins for each candidate and baseline.

Artifacts: `backtest.md`, `backtest.json`, `selection_2022_predictions.csv`, `audit_2023_2025_predictions.csv`. Raw rows include features, QB source/cutoff, training sample count and latest training availability. Fixed settings, exact package versions, source/data/prediction hashes and sample accounting are retained.

## Material limitations and publication blockers

Historical Tuesday data vintages are unavailable. This rebuild uses corrected/revised current nflverse files, Tuesday noon Eastern prediction cutoffs and a conservative final-score/kickoff-plus-12-hours availability proxy. Latest schedule actual starter IDs are ignored. Without timestamped evidence, the lagged dominant passer can miss injuries or announced changes. Both availability and observation timestamps are required for explicit QB evidence. New predictions store their resolved inputs; old picks are never backfilled.

This is a retrospective repair audit: 2023–2025 were already inspected in the flawed original build, so it is not an untouched prospective validation. Closing lines are later-information benchmarks. Ties are excluded from binary training/audit scores and remain pushes in the live record; probabilities are conditional on a decisive result.

`make publish PUBLISH_BRANCH=main` is prepared as an explicit reviewed workflow: clean checkout/index, branch/upstream/remote/project preflight, genuine new picks, artifact commit, push, staged deployment, READY/content verification, promotion and production CSV verification. No publication happened this turn. Local Vercel CLI and project link are missing; the two Git attempts failed `git_info_fail`, and the old public READY site cannot be matched to a Git SHA. Persistent credential/access changes require separate approval. See `deployment-diagnosis.md` for exact remote evidence and setup choices. Concurrent pick writers remain unsupported; run a single weekly process.

The original October 14 completion / before-October 15 first-pick deadline is unchanged; publication setup and approval remain with the parent task.
