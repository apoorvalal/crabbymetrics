# Project Agent Notes

## Public Class Docs

- Every public-facing Python class exposed from `src/lib.rs` should have a clean, self-contained docs page under `docs/examples/` or an explicitly named docs page linked from the site navigation.
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

## Bookkeeping

- Before any push, make sure `devspec.md` and `devlog.md` reflect the current branch state.
