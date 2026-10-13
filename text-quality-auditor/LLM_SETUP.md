# Hosted Hugging Face now; local Qwen later

## Current default

The CLI's default provider is `huggingface`. Default model:

```text
Qwen/Qwen3-235B-A22B-Instruct-2507:nscale
```

This calls the Hugging Face Inference Providers router at `https://router.huggingface.co/v1/chat/completions`, selecting the named provider explicitly. It does not download or run a 235B model on your laptop. The publisher describes this variant as non-thinking instruction-following. The public router listing was checked during implementation: the nscale mapping was live and reported structured-output support. Availability, permissions and billing can change.

Hugging Face structured JSON output is requested using the portable version-2 policy schema. Local validation still enforces cross-field rules: weights must sum to 1, anchors must be ordered, metrics must fit the context, and mandatory gates cannot be waived. Schema-shaped output alone does not ensure appropriate business decisions.

## Set credentials securely

Create a token at https://huggingface.co/settings/tokens with Inference Providers permission. Set `HF_TOKEN` in your local environment or secret manager. Do not send the token in chat or commit it to source control. `.env` files are ignored by Git, but the application does not automatically load them.

For a temporary Bash terminal session, enter it without echoing or putting the value into shell history:

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
```

The provider may require inference credits/billing. No payment or account configuration was changed by this project. Token validation fails before reading your dataset if `HF_TOKEN` is missing.

## Run a real hosted audit

```bash
python3 -m pip install -e '.[gx]'

python3 -m textaudit.cli data/banking77/dataset.csv \
  --label-column label --split-column split \
  --allowed-labels data/banking77/labels.json \
  --purpose supervised_training \
  --use-case "Assess data for a 77-intent banking support classifier. Prioritize complete valid labels, independent evaluation splits and sufficient per-class training support. Preserve natural short customer queries and report assumptions." \
  --provider huggingface --backend gx --timeout 300 \
  --out reports/local/huggingface-banking
```

The default model is selected automatically. Override with `--model <model-id>:<provider>` only after checking that the chosen pairing supports JSON-schema output. A failed provider request does not silently switch to another model. The explicit Hugging Face provider choice authorizes sending aggregate statistics and entered business context to its hosted router; `--allow-remote` is only needed for a custom remote compatible endpoint.

Only aggregate profile and business context enter the planner. Raw messages, label names, group IDs and dataset paths are not transmitted. GX and SQLite run locally. The business context is sent as entered, so do not put secrets in it.

`--max-tokens` defaults to 4000. One normal request and at most one invalid-policy repair request are allowed. Provider authentication, billing and rate-limit errors stop with a readable error; the provider response body and token are not logged. Truncated outputs fail rather than being treated as valid policies. The report records provider, configured model, prompt version and returned usage when available; usage is not a calculated billing total.

## Switch to local Qwen

Install Ollama and a Qwen model that fits your RAM/GPU. For example, `qwen3:8b` is an available model tag, but suitability depends on your computer and output-contract testing:

```bash
ollama pull qwen3:8b
```

With the local server running, use the same audit command and replace the hosted selection with:

```bash
--provider local --model qwen3:8b \
--base-url http://127.0.0.1:11434/v1 --timeout 300
```

The local mode uses JSON-object output plus the same application validation. It requires a loopback endpoint and does not send Hugging Face or generic API credentials. It does not start Ollama or download model weights automatically. Configure a genuinely local downloaded model rather than a cloud-forwarding server.

Qwen versions can differ in reasoning behavior and JSON compliance. Use a suitable instruction/non-thinking configuration for policy generation and test it against the policy corpus. Invalid output is rejected after one repair. The same interface is reused; a smaller local model is not guaranteed to produce the same weights as the hosted model. Hold an approved saved policy fixed when comparing dataset versions.

## Other compatible endpoints

```bash
--provider compatible --model MODEL_ID \
--base-url https://YOUR_SERVER/v1 --allow-remote
```

This provider reads `TEXTAUDIT_API_KEY` if present. The Hugging Face mode accepts only its fixed router URL so its credential cannot accidentally be sent to an arbitrary endpoint. Remote compatible mode requires HTTPS. Redirects are rejected.

## Validation status

The provider configuration, request formatting, credentials routing, strict schema, error handling, policy validation and cache behavior were tested with mocked transport and a mock local server. Great Expectations runs were tested on actual publisher datasets.

**No real authenticated Hugging Face request was made here: no `HF_TOKEN` was available.** No locally installed Qwen was available either. The integration is configured and tested at the transport-contract level; a live model smoke test is still required with your token or local server. Existing bundled reports remain clearly labelled explicit-policy demonstrations.

Primary references:

- https://huggingface.co/docs/inference-providers/tasks/chat-completion
- https://huggingface.co/docs/inference-providers/guides/structured-output
- https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507
- https://router.huggingface.co/v1/models
- https://docs.ollama.com/api/openai-compatibility
- https://ollama.com/library/qwen3:8b
