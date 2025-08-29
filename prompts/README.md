# Prompts

This folder contains prompt datasets used by the **Shadow Policy Attack (SPA) Battery**.  
They are derived from the **JailbreakBench Behaviors** dataset and prepared for direct use with `spa_runner.py`.

---

## Source & License

- **Upstream dataset:** [JailbreakBench / JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- **Authors:** JailbreakBench Team
- **Upstream license:** MIT License © 2023 JailbreakBench Team  
  See [`LICENSE`](./LICENSE) in this folder for the full text. Per MIT terms, we preserve their copyright
  and license notice with any copies and substantial portions.

---

## Files in this directory

- `JailBreakBench_Harmful.csv` — Derived CSV formatted for SPA Runner.  
  - Columns follow the runner’s expected headers (e.g., `Index`, `Goal`, `Behavior`, `Category`, `Source`).
  - See **Modifications** below for details.

You may add additional prompt files here (CSV or JSON). The runner accepts:
- **CSV** with headers as described in the main project README.
- **JSON** with top-level arrays keyed by suite (e.g., `negative`, `trigger`).

---

## Modifications made in this repository

We have adjusted a subset of prompts to reduce risk while preserving test intent:

- **CSE → adult-only:**  
  Requests originally testing for **child sexual exploitation (CSE)** behaviors were rewritten to test for **objectionable but adult-only** material.  
  This avoids generating any CSE-related content while preserving the challenge characteristics of the original prompts.

- **Attribution update in modified rows:**  
  For rows adapted from CSE-related content, the credit field now uses the format  
  `Original (based on …)` and preserves the **original request attribution** in line with the format used by the upstream dataset.

- **Minor formatting:**  
  Light cleanup to ensure CSV compatibility with the SPA Runner pipeline (no semantic changes besides the CSE → adult-only rewrite described above).

---

## How to use with SPA Runner

Run the battery against this CSV from the repository root:

```bash
python spa_runner.py \
  --suite negative \
  --prompts prompts/JailBreakBench_Harmful.csv \
  --tests all
