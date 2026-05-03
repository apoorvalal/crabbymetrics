# Project Agent Notes

## Public Class Docs

- Every public-facing Python class exposed from `src/lib.rs` should have a clean, self-contained docs page under `docs/examples/` or an explicitly named docs page linked from the site navigation.
- Do not rely on the generated API table alone as the only documentation for a public class.
- Small estimator examples should construct synthetic data in the page itself, call the public API directly, and print or plot at least one output from `summary()` / `predict()` where those methods exist.
- If one page documents multiple closely related public classes, the page title/nav text should make that grouping obvious.

## Ablation Docs

- Any Quarto page under `docs/ablations/` should set `execute.cache: true`.
- Any Quarto page under `docs/ablations/` should set `freeze: auto`.
- Keep code in ablation docs present but folded.
- Prefer using `crabbymetrics` estimators directly inside ablation docs instead of hand-coded stand-ins unless the point of the page is an explicit reference calculation.
- In Quarto docs, always use `$...$` and `$$...$$` for math delimiters. Do not use `\(...\)` or `\[...\]`.

## Bookkeeping

- Before any push, make sure `devspec.md` and `devlog.md` reflect the current branch state.
