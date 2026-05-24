# Project Agent Notes

## Public Class Docs

- Every public-facing Python class exposed from `src/lib.rs` should have a clean, self-contained docs page under `docs/examples/` or an explicitly named docs page linked from the site navigation.
- Keep `docs/llms.txt` up to date whenever public docs/navigation materially change.
- Keep the docs site's API dropdown up to date, but reserve it for first-tier public classes and API reference pages only; put vignettes, examples, ablations, and topic pages in the other navigation groups.
- Do not rely on the generated API table alone as the only documentation for a public class.
- Small estimator examples should construct synthetic data in the page itself, call the public API directly, and print or plot at least one output from `summary()` / `predict()` where those methods exist.
- If one page documents multiple closely related public classes, the page title/nav text should make that grouping obvious.

## Ablation Docs

- Any ablation belongs in a well-documented Quarto page under `docs/ablations/`, not as a loose script under `benchmarks/`.
- Any Quarto page under `docs/ablations/` should set `execute.cache: true`.
- Any Quarto page under `docs/ablations/` should set `freeze: auto`.
- Keep code in ablation docs present but folded.
- Prefer using `crabbymetrics` estimators directly inside ablation docs instead of hand-coded stand-ins unless the point of the page is an explicit reference calculation.
- In Quarto docs, always use `$...$` and `$$...$$` for math delimiters. Do not use `\(...\)` or `\[...\]`.

## Rendered Review Previews

- When a docs-heavy PR adds or materially changes a Quarto page, render the docs locally and copy the whole rendered docs site to the Hetzner draft host so navigation/assets can be reviewed before merge.
- Use `QUARTO_PYTHON=.venv/bin/python quarto render <page.qmd>` for targeted Python-backed Quarto renders, or render the relevant docs set when navigation/search changed.
- Copy the rendered site with `rsync -az --delete --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r docs/ hetz:/root/lalten/drafts/<descriptive-name>/`, then ensure files are world-readable if needed.
- Share the preview URL as `https://lalten.org/drafts/<descriptive-name>/` and, when helpful, the specific page URL under that directory.

## Public Docs Deployment

- The public docs URL is `https://apoorvalal.github.io/crabbymetrics/`.
- GitHub Pages serves from the root of the `gh-pages` branch, not directly from `master`.
- **No rendered HTML on `master`:** do not commit rendered `.html`, Quarto cache/freeze artifacts, or `docs/search.json` changes to `master` / source PR branches. Source branches should carry `.qmd` and code/source changes only.
- Before deploying or previewing, render the affected page(s) locally; if navigation/search changed, render enough of the site to update the generated site tree:
  - `QUARTO_PYTHON=.venv/bin/python quarto render docs/examples/<page>.qmd`
  - or `QUARTO_PYTHON=.venv/bin/python quarto render docs` for broad nav/search changes.
- Use rendered outputs only for local checking, Hetzner review previews, or the `gh-pages` branch.
- To deploy, copy the rendered `docs/` tree to a fresh `gh-pages` clone and push that branch. Use a clone, not a worktree: `rsync --delete` can delete a worktree's `.git` file.

```bash
TMPDIR=$(mktemp -d)
git clone --branch gh-pages --single-branch git@github.com:apoorvalal/crabbymetrics.git "$TMPDIR"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.quarto/' \
  --exclude '**/.jupyter_cache/' \
  --exclude '**/*.quarto_ipynb' \
  docs/ "$TMPDIR"/
touch "$TMPDIR/.nojekyll"
(cd "$TMPDIR" && git add -A && git commit -m "Publish rendered docs site" && git push origin gh-pages)
rm -rf "$TMPDIR"
```

- If the `gh-pages` commit has no changes, there was nothing new to deploy. Otherwise, check the Pages deployment in GitHub Actions and spot-check the public URL after it finishes.

## Release Packaging

- **Never ship rendered docs in the PyPI sdist.** A `v0.6.6` release attempt tried to upload a 100+ MB source tarball because rendered Quarto docs were included in the sdist; PyPI rejected it. Keep `docs/**/*` excluded from sdists in `pyproject.toml` (`[tool.maturin].exclude`) and verify with `maturin sdist --out /tmp/...` before tagging releases when docs changed. A normal sdist should be small, not tens/hundreds of MB.

## Bookkeeping

- Before any push, make sure `devspec.md` and `devlog.md` reflect the current branch state.
