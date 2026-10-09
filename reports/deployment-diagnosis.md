# Vercel deployment diagnosis — 2026-10-09

Read-only inspection performed on 2026-10-09 at approximately 01:14–01:18 UTC. No deployment, push, project connection, access-setting change, credential change, or authentication-bypass link was created. This report records deployment evidence separately from the model repair and backtest results.

## Confirmed remote state

| Item | Observed value |
| --- | --- |
| Team | `damianlujan1s-projects` / `team_RNPL71TWtCTxnPNal4E8dGvM` |
| Project | `gridiron-model` / `prj_ewgxRzaTU7Kga2OzkRco28m7TFnp` |
| Latest attempt | `dpl_D9kXfn3achHAgqyHpeDsWmZZBT3L`, `ERROR`, `git_info_fail` |
| Previous attempt | `dpl_AaPWayNEvHncgqGCchir264UDdLW`, `ERROR`, `git_info_fail` |
| Git metadata on both failed attempts | `DamianLujan1/coolshit`, branch `main`, SHA `5caa7a732942721e3822a035b727d52902c11ab1` |
| Production deployment | `dpl_BhFENzi2MNZChxBYdbMpsEPsp38E`, `READY`, target `production` |
| Production source provenance | Empty `meta` and no returned `gitSource`, even when requesting Git repository information; its Git SHA is unverified |
| Production alias | `gridiron-model.vercel.app` still points to `dpl_BhFENzi2MNZChxBYdbMpsEPsp38E` |
| Domain | `gridiron-model.vercel.app`, verified, no redirect |
| Vercel Authentication | Enabled for `prod_deployment_urls_and_all_previews` |
| Password / trusted-IP protection | Both disabled; left unchanged |

The [latest failed attempt](https://vercel.com/damianlujan1s-projects/gridiron-model/D9kXfn3achHAgqyHpeDsWmZZBT3L) returned an empty build-events array. Failure happened approximately 2.2 seconds after creation, before any returned build log. The previous attempt shows the same error. This supports a Git-source acquisition problem, not an observed application build failure; the exact underlying integration/access failure is not exposed by these responses.

`get_git_deployment_context` lists three linked projects on this team: two for `DamianLujan1/mvp1midland` and one for `DamianLujan1/satellite-decay`. It lists no `gridiron-model` or `coolshit` Git connection. `list_projects` filtered by `repoUrl=https://github.com/DamianLujan1/coolshit` returns zero projects. Git metadata on a deployment is therefore not evidence that this project is Git-linked.

Repository discovery through Vercel's documented `search_repo` tool failed with HTTP 400, `bad_request`, `Search request invalid`, both with `provider=github` and with provider inference. No available namespace-discovery tool supplied the missing integration context. These failed reads do **not** prove that the GitHub App lacks permission; repository installation scope remains unverified.

## Public access and published content

The local sandbox initially could not resolve the hostname. A subsequently approved, read-only network request succeeded:

```text
curl -I -L --max-time 25 https://gridiron-model.vercel.app
HTTP/2 200
date: Fri, 09 Oct 2026 01:15:55 GMT
content-type: text/html; charset=utf-8
content-length: 6380
last-modified: Wed, 07 Oct 2026 22:18:49 GMT
x-vercel-cache: HIT
```

`curl -sS --max-time 25 https://gridiron-model.vercel.app` also succeeded. The page title is `Week 5 picks, 2026 | Gridiron Model`; the table contains all 15 Week 5 picks and uses city/abbreviation names. Its footer contains nflverse attribution, the entertainment/not-betting-advice text, and `Model v0.1.0, generated 2026-10-07 21:42 UTC`. This confirms the old production page remains reachable; it does not establish that the repaired model has been published.

## Local inspection

The repair checkout is `/Users/damianlujan/Documents/Codex/2026-10-08/task/coolshit`, branch `repair/causal-qb-weekly-safety`. It has no `.vercel/project.json`. The root `vercel.json` sets `framework`, `buildCommand`, and `installCommand` to null and `outputDirectory` to `site`; `site/vercel.json` configures clean URLs and no trailing slash.

At inspection, `command -v vercel` returned exit 1: no Vercel CLI is available on the current PATH. The system Git shim returned the Xcode-license message; read-only status succeeded with `/Library/Developer/CommandLineTools/usr/bin/git`. No license, credential, or shell configuration was changed.

## Publication plan for a later authorized run

The explicit weekly publication workflow should reject synthetic `--now`, verify the intended repository/branch and preserved historical pick bytes, run the data/model/site checks, commit the resulting genuine picks with their real creation timestamps, push that exact commit, publish the exact generated site to the existing project, and verify the resulting deployment and production alias. It must stop on a failed push or deployment. Local commit alone does not publish the audit trail or the site.

For the currently unlinked project, a CLI/static-file deployment can avoid the failing Git-source path. First install a reviewed, pinned Vercel CLI and let the user authenticate directly. Then, only with the user's approval for setup/publication, link this directory to the **existing** project (not a newly created project):

```bash
vercel link --project gridiron-model --scope damianlujan1s-projects
```

Validate that the resulting `.vercel/project.json` contains exactly the project and team IDs above before permitting the publish command. Do not print tokens or write them into repository files. With reviewed generated files committed and pushed, a staged production sequence from the repository root is:

```bash
vercel deploy --prod --skip-domain --scope damianlujan1s-projects \
  --meta sourceCommit=REVIEWED_COMMIT_SHA
vercel inspect DEPLOYMENT_URL --scope damianlujan1s-projects
# Confirm READY, intended project/team, commit metadata, and the site's contents.
vercel promote DEPLOYMENT_URL --scope damianlujan1s-projects
curl -I -L --max-time 25 https://gridiron-model.vercel.app
```

These are a reviewable future sequence, **not commands executed during this repair**. `REVIEWED_COMMIT_SHA` and `DEPLOYMENT_URL` are required resolved values. A production URL may require authorized testing because the existing Vercel Authentication setting protects deployment URLs; do not disable that protection to test it. Metadata records provenance but does not connect a repository or guarantee that uploaded files match the SHA. The publishing implementation must verify clean source state and upload only the reviewed site output.

Alternatively, the user can authorize connecting the existing project to `DamianLujan1/coolshit` with `vercel git connect` after local project linking. If Vercel requests GitHub App repository access, that persistent access grant needs the user's approval. Before enabling Git-driven publication, confirm the intended production branch and static `site` output settings; a push can then trigger a deployment. Avoid combining automatic Git deploys with a second unconditional CLI deploy.

The original `vercel deploy --prod --yes site` command from an unlinked checkout is insufficient: it lacks a verified existing project/team target and a pushed-commit gate. The repaired explicit workflow should fail closed until its deployment prerequisites are satisfied.

## Remaining blockers

- Publication is explicitly out of scope for this repair turn; nothing was pushed or deployed.
- The current project has no verified Git connection to `coolshit`; Git-source deployment has failed twice. Exact integration cause remains unverified.
- The local checkout has no Vercel project link or available CLI. Setting up persistent credentials, granting GitHub access, or altering remote project/security settings requires separate user approval.
- The old READY deployment cannot be tied to a Git SHA through the available metadata.
- End-to-end publication cannot be claimed tested until a later authorized push, deploy, and public verification succeed. Mocked command tests may verify workflow gates only.

Official references consulted: [Vercel Git CLI](https://vercel.com/docs/cli/git), [link](https://vercel.com/docs/cli/link), [deploy options](https://vercel.com/docs/cli/deploy), [inspect](https://vercel.com/docs/cli/inspect), and [promote](https://vercel.com/docs/cli/promote).
