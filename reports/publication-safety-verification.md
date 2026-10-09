# Publication safety verification — 2026-10-09

The publication wrapper no longer calls `vercel curl` or any credential, bypass-token, login, refresh or protection mutation command. It permits a fixed set of reviewed CLI 63.1.0 commands, requires an explicitly supplied existing token, uses isolated global configuration, refuses anonymous credential state, and sanitizes command output. No saved CLI credential is used. Public verification reads exact bytes using unauthenticated HTTPS without cookies, proxies, redirects or an authentication fallback. Protected previews are recorded as unverified and stop CLI promotion.

The separate, explicitly authorized source-files deployment uses the existing connected Vercel tool and existing project. It does not use the saved expired CLI credential. Only reviewed static site files and deployment configuration are eligible for upload. The local commit is recorded as local-only source provenance; no Git push or merge is authorized.

## Validation

- `.venv/bin/python -m pytest`: 144 passed.
- `git diff --check`: clean.
- `git diff --exit-code 5caa7a732942721e3822a035b727d52902c11ab1 -- data/picks site/picks.csv recaps`: no changes.
- Independent source review: no blocking credential or bypass findings; seven intended regular site files, no symlinks, caches, credentials or private paths.
- Canonical and site picks CSV SHA-256: `a5039d5c2070268e663537a1daad8a3c0a7c81a2caf4a3e8c44fd95b808c39a5` (15 original issued picks).
- Public backtest is byte-identical to `reports/backtest.md`: `374db3872788cb1ca1a6f42f20baaf6a75259f7c4eef9f168857bc3f96fb99a5`.

The causal model and historical audit from local repair commit `015ceeaf8684be607930d70053509989413c920f` are unchanged by this publication correction. Their reproducibility manifest describes that evaluation snapshot. Publication verification is tested here; terminal deployment state and public HTTP evidence will be recorded separately after deployment.
