# Week 6 delivery verification — October 15, 2026

The real online command `.venv/bin/python -m gridiron weekly --season 2026 --week 6 --no-commit` refreshed nflverse schedules and 2026 play-by-play, team and player statistics, validated them, trained the selected logistic model on 3,096 prior decisive games, appended 14 picks, graded Week 5, and generated the record, recap and static site. No synthetic clock was supplied.

All 14 new picks were issued at **2026-10-15 13:24:59 UTC**, before the earliest kickoff at **2026-10-16 00:15 UTC**. The log now has 29 unique games. Its original 2,295-byte prefix (all 15 Week 5 picks) is unchanged: SHA-256 `a5039d5c2070268e663537a1daad8a3c0a7c81a2caf4a3e8c44fd95b808c39a5`. Full log SHA-256: `5646c622c422245ee22b9281ef12719d67499c5dbab88c33ed95945f3b3d1388`.

## Results

| Week 5 | Wins | Losses | Pushes | Brier |
| --- | ---: | ---: | ---: | ---: |
| Original model picks (v0.1.0) | 10 | 4 | 1 | 0.2186 |
| Always home | 5 | 9 | 1 | — |
| Closing-spread favorite | 9 | 5 | 1 | 0.2275 |
| Moneyline favorite / no-vig probabilities | 9 | 5 | 1 | 0.2267 |

All 15 Week 5 games are graded; 14 Week 6 picks are pending. The tie is a push and excluded from binary accuracy/Brier. Week 5's model accuracy is 10/14 (71.43%) versus 9/14 (64.29%) for the moneyline favorite. These are the original v0.1.0 picks; they were not recomputed or relabeled as repaired predictions.

This small live sample does not establish superiority. The unchanged reconstructed 2023–2025 audit did **not** beat the market: logistic accuracy/Brier/log loss 63.23%/0.2230/0.6364, no-vig market 67.92%/0.2102/0.6077. Logistic was selected using 2022 alone. Historical files are rebuilt current nflverse releases, not archived prediction-day vintages; closing lines are later-information benchmarks. See `../backtest.md`.

## Verification

- `.venv/bin/python -m pytest`: **146 passed** after the provenance clarification regression.
- `.venv/bin/python -m gridiron validate`: passed offline after fresh downloads; schedules/PBP/team/player retrievals all dated October 15, 13:24:57–59 UTC. No cache fallback occurred. Details and hashes are in `data-validation.json`.
- `.venv/bin/python -m compileall -q gridiron`: passed.
- Git whitespace check passed with `core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol`. The ordinary check flags the existing CSV CRLF format; those line endings are intentionally preserved.
- Independent calculations matched every Week 5 graded hit and Brier score, full Week 6 schedule coverage, the 28 lagged QB IDs and their completed prior-game PBP sources, pre-kickoff timestamps, and exact static asset copies.
- All site files are intended public static assets, with visible nflverse attribution and the entertainment/not-betting-advice footer. No cache, credentials, symlinks or private paths are included.

## QB provenance clarification

The initial Week 6 sidecar accidentally retained target-schedule QB name metadata alongside the correctly lagged IDs. Seattle's schedule name was Drew Lock while the causal ID represented Sam Darnold. Explicit numeric model feature selection never used those names, so probabilities and picks are unaffected.

The issued CSV and original `qb_inputs.jsonl` bytes are preserved. `qb_input_annotations.jsonl` contains 14 timestamped append-only clarifications excluding both schedule name fields; the causal IDs, source games, availability timestamps and numeric estimates remain authoritative. Future serialization now allowlists only canonical causal fields. Two new regressions change names to other strings or null and confirm identical saved causal provenance.

## Repository and publication scope

The initial checkout was clean at `3c2475d46957947b95b714a67e2fc199cfcad36a`. Fresh remote inspection found main still at `5caa7a732942721e3822a035b727d52902c11ab1` and a separate Claude branch containing a training-window experiment but no new picks. That branch is preserved; it is not merged into this verified repaired model. Publication targets the repair branch without a merge or force-push, and the existing `gridiron-model` Vercel project through the connected source-files route.

This committed report records pre-publication validation. Remote commit identity and live public HTTP hashes are verified separately after publication; no successful remote write is implied by this file alone.

Data: [nflverse](https://github.com/nflverse), loaded with nflreadpy, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). FTN and participation datasets are excluded. For entertainment only; not betting advice.
