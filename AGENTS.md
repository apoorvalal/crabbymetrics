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

- When a docs-heavy PR adds or materially changes a Quarto page, render the page locally and copy the self-contained HTML to the Hetzner draft host so it can be reviewed before merge.
- Use `QUARTO_PYTHON=.venv/bin/python quarto render <page.qmd>` when rendering Python-backed Quarto docs from this repo.
- Copy review HTML with `scp <rendered.html> hetz:/root/lalten/drafts/<descriptive-name>.html`, then ensure it is world-readable on the server (`chmod 644`).
- Share the preview URL as `https://lalten.org/drafts/<descriptive-name>.html`.

## Bookkeeping

- Before any push, make sure `devspec.md` and `devlog.md` reflect the current branch state.
