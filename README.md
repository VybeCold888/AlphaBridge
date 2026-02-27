# AlphaBridge - Codespaces Quickstart

This README is for users you share this project with. It explains only how to run the web app in GitHub Codespaces.

## 1) Open in Codespaces

1. Open this repository on GitHub.
2. Click **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for setup to finish (dependencies install automatically).

## 2) Add your Hugging Face token (required)

The app uses a hosted open-source model and needs your own token.

1. In GitHub, go to this repo: **Settings** → **Secrets and variables** → **Codespaces**.
2. Click **New repository secret**.
3. Name: `HF_TOKEN`
4. Value: your Hugging Face token

Then in the Codespace terminal run:

```bash
export LLM_BACKEND=huggingface
export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export LLM_ALLOW_OFFLINE_FALLBACK=false
```

## 3) Start the app

From the project root:

```bash
./.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## 4) Open the web app

1. Open the **Ports** panel in VS Code.
2. Find port **8000**.
3. Click **Open in Browser**.
4. Use the app in that tab.

## If it doesn’t work

- Check health endpoint: open `/health` on the same URL and confirm `"backend":"huggingface"` and `"ok":true` under `llm_connection`.
- If unauthorized errors appear, recheck the `HF_TOKEN` secret value.
