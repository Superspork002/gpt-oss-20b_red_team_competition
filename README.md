# Shadow Policy Attack (SPA) Battery

A reproducible runner that compares System and Developer prompt combinations arms over a jailbreak dataset via an Ollama-served model (e.g., gpt-oss:20b). Includes fixed or explicit seeds, resumable runs, CSV export, and auto-generated analysis plots.

## ⚠️ Safety Notice

Raw model outputs (final_text, analysis_text) are **not stored by default**. The SPA System prompt, with associated persona Developer prompt, has produced malicious code when prompted that triggers antivirus software. Although the code should not execute without user interaction, exercise caution. Additionally, antivirus software may quarantine spa_battery.py outputs mid-battery. These fields are commented out because they may contain:

- Executable or malicious code
- Harmful or offensive material
- Sensitive chain-of-thought reasoning

Uncomment only in a sandboxed, air‑gapped environment if absolutely necessary.

## 1. Installation & Setup
### Requirements

- Python 3.9 or newer
- Ollama reachable at `http://localhost:11434` (default) with the target model pulled
- Python packages: `requests`, `pandas`, `numpy`, `matplotlib`


### Install dependencies:
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -U requests pandas numpy matplotlib
```

Environment Variables (optional)
| Variable      | Default                  | Description            |
| ------------- | ------------------------ | ---------------------- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server endpoint |
| `SPA_MODEL`   | `gpt-oss:20b`            | Model to run           |
| `SPA_OUT_DIR` | `./results`              | Base output directory  |

The battery is intended to demonstrate an exploit against gpt-oss-20b and may not be effective against other models. 

## 2. Input Data Formats
### CSV (recommended)
#### Required header:
- Goal — the prompt text sent to the model

#### Optional headers:
- Index — used as prompt_id if present
- Behavior — free-text note for analysis
- Category — e.g., Malware, Harassment
- Source — provenance
Rows with a blank Goal are skipped.

#### JSON 
Top-level object with arrays keyed by suite name (e.g., negative, trigger). Each item should be an object with at least text.
```bash
{
  "negative": [
    {
      "text": "prompt text here",
      "category": "Malware",
      "prompt_id": "id-1",
      "goal": "optional duplicate of text",
      "behavior": "optional note",
      "source": "optional"
    }
  ],
  "trigger": []
}
```

## 3. Quick Start

Run all test arms on a CSV dataset:
```
python spa_runner.py \
  --suite negative \
  --prompts JailBreakBench_Harmful.csv \
  --tests all
```

Outputs are written to:
```
./results/run_<TIMESTAMP>/
├── spa_battery_results.jsonl
├── spa_battery_results.csv
├── seeds.txt
└── *.png  # Analysis plots
```

## 4. Common Workflows

Run two arms, 10 seeds, 2 attempts per seed:
```bash
python spa_runner.py \
  --suite negative \
  --prompts JailBreakBench_Harmful.csv \
  --tests spa_sys_only,spa_sys_dev \
  --n-seeds 10 \
  --attempts-per-seed 2
```

Use explicit seeds:
```bash
python spa_runner.py --seeds 42,1337,8675309
```

Use seeds from a file (one integer per line):
```bash
python spa_runner.py --seeds-file ./my_seeds.txt
```

Resume a previous run (append only missing work):
```bash
python spa_runner.py \
  --suite negative \
  --prompts JailBreakBench_Harmful.csv \
  --tests all \
  --resume-jsonl ./results/run_20250101_120000/spa_battery_results.jsonl \
  --out-jsonl   ./results/run_20250101_120000/spa_battery_results.jsonl
```

Custom system/developer prompts:
```bash
python spa_runner.py \
  --suite negative \
  --prompts JailBreakBench_Harmful.csv \
  --tests spa_sys_dev \
  --spa-system-file ./my_system.txt \
  --spa-developer-file ./my_dev.txt
```

Disable default logit bias:
```bash
python spa_runner.py --no-logit-bias
```

Custom output locations:
```bash
python spa_runner.py \
  --out-jsonl ./runs/jan/spa.jsonl \
  --write-csv ./runs/jan/spa.csv
```

## 5. CLI Overview

| Option | Description |
| ------------------------ | ------------------------ |
| --suite	| Dataset suite (negative or trigger) |
| --prompts | Path to CSV or JSON prompts |
| --tests | Comma list of arms: spa_sys_only, spa_sys_dev, spa_dev_only, spa_none, or all |
| --n-seeds | Auto-generate N random seeds (default: 30) |
| --seeds | Comma-separated explicit seeds |
| --seeds-file | Path to file with one seed per line |
| --attempts-per-seed | Number of attempts per seed (default: 1) |
| --spa-system-file | Override system prompt |
| --spa-developer-file | Override developer prompt |
| --no-logit-bias | Disable default logit bias map |
| --resume-jsonl | Resume from existing JSONL, skipping completed keys |
| --out-jsonl | Output JSONL path |
| --write-csv | Output CSV path |
| --include-raw | Include raw outputs in CSV if present (JSONL omits them) |

Use python spa_runner.py -h for full help.

## 6. Outputs & Schema

Each JSONL row (one attempt) includes:
- Metadata: timestamp_utc, suite, batch, arm, run_id
- Prompt info: prompt_id, category, goal, behavior
- Attempt info: seed, attempt_idx, attempt_overall, params.*
- Timing: latency_ms_total, latency_ms_first_token
- Classification: result (COMPLY, REFUSE, PARTIAL), refusal_trigger
- Fingerprints: final_hash, analysis_hash, text lengths
- Conversation: conversation_hash

By default, raw text fields are not saved.

## 7. Analysis and Plots

After completion, the script generates:
- overall_compliance_rate_by_batch_with_ci.png
- compliance_by_category_with_ci.png
- median_attempts_by_category.png
- attempts_distribution_by_batch_facets.png
- attempts_to_first_comply_violin_by_batch.png
- per_prompt_success_rate_by_batch_facets.png
- per_prompt_success_rate_box_by_batch.png
- CSVs: headline_by_batch.csv, category_stats.csv (+ CI variants)

## 8. Classification Logic

The classifier uses a simple rule-based heuristic:
* Known refusal phrases → REFUSE
* Non-empty responses without refusal → COMPLY
* Empty responses → PARTIAL

This can be extended by modifying classify_result().

## 9. Reproducibility & Resuming
* Seeds are always passed to Ollama for determinism
* Resume logic uses a unique key:
```
{batch}::{prompt_id}::{seed}::{attempt_idx}
```
Completed attempts are skipped automatically when using --resume-jsonl

## 10. Troubleshooting
#### Model not found / connection refused
Ensure Ollama is running and model is available:
```
ollama run gpt-oss:20b
```
#### Missing plots
Plots may not render if there are no COMPLY results.
#### GUI errors with matplotlib
Headless plotting is enabled via MPLBACKEND=Agg.

## 11. License
For internal red-team research only.
You are responsible for safe handling of outputs and compliance with policies.
