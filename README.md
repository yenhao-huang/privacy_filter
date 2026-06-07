# privacy_filter

`privacy_filter` is a small dependency-free Python project for detecting and redacting common sensitive text before it is logged, stored, or sent to downstream systems.

## Layout

- `core/service/`: privacy detection and redaction logic
- `core/api/`: command-line entry points
- `configs/`: default pattern configuration
- `test/`: standard-library test suite
- `lib/`: shared helpers

## Quick Start

Run the CLI from the project root:

```bash
python -m core.api.cli "Contact jane.doe@example.com at 555-123-4567"
```

Run tests:

```bash
python -m unittest discover -s test
```

## Configuration

Default regex patterns live in `configs/privacy_patterns.json`. The service has built-in patterns, so the config file is optional at runtime.

## Score

### masked string exact match

This metric creates the gold masked string from benchmark spans, removes whitespace from gold and prediction, then gives each row score `1` for exact match or `0` otherwise.

| model | total | correct | score |
| --- | ---: | ---: | ---: |
| OpenAI privacy-filter | 160 | 54 | 0.338 |
| qwen3.6-35b-a3b vLLM-compatible | 160 | 116 | 0.725 |
| gemma4-26b-a4b vLLM-compatible | 160 | 115 | 0.719 |
| Presidio custom recognizers | 160 | 59 | 0.369 |

OpenAI masked string exact match by label:

| label | total | correct | score |
| --- | ---: | ---: | ---: |
| `account_number` | 20 | 4 | 0.200 |
| `private_address` | 20 | 10 | 0.500 |
| `private_date` | 20 | 2 | 0.100 |
| `private_email` | 20 | 13 | 0.650 |
| `private_person` | 20 | 7 | 0.350 |
| `private_phone` | 20 | 10 | 0.500 |
| `private_url` | 20 | 5 | 0.250 |
| `secret` | 20 | 3 | 0.150 |

qwen3.6-35b-a3b masked string exact match by label:

| label | total | correct | score |
| --- | ---: | ---: | ---: |
| `account_number` | 20 | 14 | 0.700 |
| `private_address` | 20 | 15 | 0.750 |
| `private_date` | 20 | 11 | 0.550 |
| `private_email` | 20 | 16 | 0.800 |
| `private_person` | 20 | 12 | 0.600 |
| `private_phone` | 20 | 17 | 0.850 |
| `private_url` | 20 | 14 | 0.700 |
| `secret` | 20 | 17 | 0.850 |

gemma4-26b-a4b masked string exact match by label:

| label | total | correct | score |
| --- | ---: | ---: | ---: |
| `account_number` | 20 | 15 | 0.750 |
| `private_address` | 20 | 16 | 0.800 |
| `private_date` | 20 | 10 | 0.500 |
| `private_email` | 20 | 16 | 0.800 |
| `private_person` | 20 | 11 | 0.550 |
| `private_phone` | 20 | 16 | 0.800 |
| `private_url` | 20 | 14 | 0.700 |
| `secret` | 20 | 17 | 0.850 |

Presidio masked string exact match by label:

| label | total | correct | score |
| --- | ---: | ---: | ---: |
| `account_number` | 20 | 9 | 0.450 |
| `private_address` | 20 | 9 | 0.450 |
| `private_date` | 20 | 4 | 0.200 |
| `private_email` | 20 | 16 | 0.800 |
| `private_person` | 20 | 3 | 0.150 |
| `private_phone` | 20 | 9 | 0.450 |
| `private_url` | 20 | 8 | 0.400 |
| `secret` | 20 | 1 | 0.050 |

### dataset

Benchmark file:

`data/benchmark/tw-pii-openai-in-shcema.csv`

The benchmark contains 160 short Traditional Chinese / Taiwan PII samples: 8 in-schema OpenAI privacy labels, 20 rows per label.

Labels:

- `account_number`
- `private_address`
- `private_date`
- `private_email`
- `private_person`
- `private_phone`
- `private_url`
- `secret`

## how to use

1. `predict_by_openai_privacy_model`

```bash
.venv/bin/python -m core.api.predict_openai_privacy_filter \
  --model-path /models/openai-privacy-filter \
  --input data/benchmark/tw-pii-openai-in-shcema.csv \
  --output results/tw-pii-openai-in-shcema_predictions.csv
```

2. `predict_by_vllm_compatible`

```bash
.venv/bin/python -m core.api.predict_by_vllm_compatible \
  --base-url http://192.168.1.78:3132 \
  --prompt-file configs/direct_replace.txt \
  --input data/benchmark/tw-pii-openai-in-shcema.csv \
  --output results/tw-pii-vllm-compatible_masked.csv
```

3. `predict_span_by_vllm_compatible`

This uses `configs/span_predict.txt` and writes normalized JSON with only `start`, `end`, and `label` for each predicted span.

```bash
.venv/bin/python -m core.api.predict_span_by_vllm_compatible \
  --base-url http://192.168.1.78:3132 \
  --prompt-file configs/span_predict.txt \
  --input data/benchmark/tw-pii-openai-in-shcema.csv \
  --output results/tw-pii-vllm-compatible_span_predictions.csv
```

4. `predict_by_presidio`

This uses Presidio custom recognizers for Taiwan benchmark patterns and writes `masked_text`.

```bash
.venv/bin/python -m core.api.predict_by_presidio \
  --input data/benchmark/tw-pii-openai-in-shcema.csv \
  --output results/tw-pii-presidio_masked.csv
```

5. `eval_by_masked_str_match`

This creates the gold masked string from benchmark spans, removes whitespace from gold and prediction, then gives each row score `1` for exact match or `0` otherwise.

```bash
.venv/bin/python -m core.api.eval_by_masked_str_match \
  --input results/tw-pii-vllm-compatible_masked.csv
```

For Presidio masked-text predictions:

```bash
.venv/bin/python -m core.api.eval_by_masked_str_match \
  --input results/tw-pii-presidio_masked.csv \
  --overall-output results/tw-pii-presidio_masked_str_match_overall.csv \
  --by-label-output results/tw-pii-presidio_masked_str_match_by_label.csv \
  --details-output results/tw-pii-presidio_masked_str_match_details.csv
```

For OpenAI token-label predictions:

```bash
.venv/bin/python -m core.api.eval_by_masked_str_match \
  --source openai_token_labels \
  --model-path /models/openai-privacy-filter \
  --input results/tw-pii-openai-in-shcema_predictions.csv \
  --overall-output results/tw-pii-openai_masked_str_match_overall.csv \
  --by-label-output results/tw-pii-openai_masked_str_match_by_label.csv \
  --details-output results/tw-pii-openai_masked_str_match_details.csv
```

## UI

1. `online_privacy_protect`

Use this Gradio UI for ad-hoc text prediction and masking. The UI supports `openai` and `qwen` engines.

```bash
.venv/bin/python ui/online_privacy_protect.py \
  --model-path /models/openai-privacy-filter \
  --host 0.0.0.0 \
  --port 7860
```

Open in browser:

`http://localhost:7860`

For qwen3.6-35b-a3b mode, set these fields in the UI:

- `engine`: `qwen`
- `Qwen base URL`: `http://192.168.1.78:3132`
- `Prompt file`: `configs/direct_replace.txt`

2. `eval_viewer`

Use this Gradio UI to inspect benchmark rows. It shows input text, ideal masked text/spans, OpenAI output, and qwen3.6-35b-a3b output on the same page. It supports `label_type` filtering plus Previous/Next navigation.

```bash
.venv/bin/python ui/eval_viewer.py \
  --model-path /models/openai-privacy-filter \
  --predictions results/tw-pii-openai-in-shcema_predictions.csv \
  --qwen-predictions results/tw-pii-vllm-compatible_masked.csv \
  --host 0.0.0.0 \
  --port 7861
```

Open in browser:

`http://localhost:7861`

## eval speed

Benchmark file: `logs/eval_speed_benchmark.csv`

Method: 5 full CLI runs on 160 rows. Timings include Python startup, CSV read/write, and for OpenAI token-label evaluation, tokenizer loading and token-label to masked-string conversion.

| method | avg sec | rows/sec | relative |
| --- | ---: | ---: | ---: |
| qwen3.6-35b-a3b `masked_text` | 1.817780 | 88.02 | 1.00x |
| Presidio `masked_text` | 1.839553 | 86.98 | 1.01x slower |
| OpenAI `openai_token_labels` | 3.144655 | 50.88 | 1.73x slower |

qwen3.6-35b-a3b and Presidio masked-string evaluation is faster because predictions already contain `masked_text`. OpenAI evaluation has to load the tokenizer and convert token labels to spans before producing the masked string.
