# Reproducibility guide

This repository separates a fully public behavioral reproduction from the
original artifact-dependent internship pipeline. The distinction is deliberate:
the bundled demo can be executed and tested by anyone, while the private FAQ
files and original trained checkpoints are neither reconstructed nor presented
as public experimental evidence.

## Reproduction scope

| Layer | Public inputs | Expected result | Status |
|---|---|---|---|
| Deterministic CLI demo | Python 3.10+ and `data/sample_faq.json` | Alias routing, intent routing, context carry-over, scoped FAQ retrieval, and safe fallback | Fully reproducible |
| Flask demo and API | Locked dependencies in `requirements-demo.lock` | Browser UI and `/api/chat` responses backed by the same synthetic fixture | Fully reproducible and CI-tested |
| Synthetic intent training | Template generator and public Python packages | A new TF-IDF + logistic-regression classifier | Re-runnable; generated model may vary with library/platform details |
| Synthetic BIO training | Generated BIO examples and `bert-base-chinese` | A newly fine-tuned token classifier | Re-runnable with network access and suitable compute; not bitwise reproducible |
| Original internship system | Private FAQ files and original trained artifacts | Historical end-to-end behavior | Not publicly reproducible because those artifacts are excluded |

The public demo reproduces control flow, not the original model outputs or an
accuracy claim. Its FAQ text is visibly marked as synthetic and intentionally
contains no current dates or URLs.

## 1. Clone and verify the dependency-free CLI

```bash
git clone https://github.com/Kathryn1206/bio-ner-vertical-qa.git
cd bio-ner-vertical-qa
python -m exam_chat_core.demo --self-test
```

Expected final line:

```text
Demo self-test passed: direct routing, context carry-over, and safe fallback.
```

Inspect a structured routing trace:

```bash
python -m exam_chat_core.demo --query "二建什么时候报名？" --json
```

The JSON should report:

- `mode` as `demo`;
- `exam` as `二级建造师考试`;
- `intent` as `registration_time`;
- a matched question and an answer marked `【演示数据】`.

The CLI imports no third-party package and does not download a model.

## 2. Reproduce the Flask interface and API

Create an isolated environment.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement requirements-demo.lock
EXAM_CHAT_MODE=demo AUTO_OPEN_BROWSER=false python web/web_app.py
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --requirement requirements-demo.lock
$env:EXAM_CHAT_MODE = "demo"
$env:AUTO_OPEN_BROWSER = "false"
python web/web_app.py
```

In another terminal, call the API:

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"二建什么时候报名？"}'
```

The response includes `"success": true`, `"mode": "demo"`, and a synthetic
answer. The HTML interface uses browser-native JavaScript and has no CDN or
other runtime web dependency.

## 3. Run the automated checks

With the locked demo dependencies installed:

```bash
python -m compileall -q exam_chat_core experiments web tests
python -m unittest discover -s tests -v
```

GitHub Actions runs the same CLI, pipeline, UI, and API checks on Python 3.10
and 3.12 for every push and pull request.

## 4. Use a different synthetic fixture

Copy `data/sample_faq.json`, keep `schema_version` set to `1`, and point the
demo backend to the new file:

```bash
DEMO_FAQ_PATH=/absolute/path/to/your_fixture.json \
python -m exam_chat_core.demo --query "二建报名条件是什么？"
```

Do not add private, personal, or proprietary data to a public fork.

## 5. Run the original full pipeline

The full backend is opt-in:

```bash
EXAM_CHAT_MODE=full python web/web_app.py
```

It requires the separately supplied BIO checkpoint, intent classifier, FAQ
workbooks, and Qwen files documented in the main README. A successful demo run
does not imply that these excluded artifacts are present.

## Determinism and provenance

- The demo fixture is authored for this repository and is not extracted from
  the original business FAQ files.
- Routing is deterministic: it performs no sampling, model download, system
  clock lookup, or network request.
- Responses avoid real dates and URLs so the fixture cannot silently become
  stale policy advice.
- The dependency-free CLI is the reference smoke test. The Flask lock file
  pins every direct and transitive Python dependency used by the web demo.
- The Git commit and Python version should be reported alongside any reproduced
  output.

## What is not claimed

This release does not claim that the original internship metrics, model
weights, or private-data behavior are reproducible. No benchmark result is
reported. Reproducing research metrics would additionally require a released,
independently annotated evaluation set and a documented evaluation protocol.
