# Deploying Evidence AI

Evidence AI runs in two inference modes. What you can deploy, and where,
depends on which mode you need.

| | Offline mode | Online mode |
| --- | --- | --- |
| Generation | Local Ollama (`qwen2.5vl:3b`) | Cloud API |
| Needs Ollama | **Yes** | **No** |
| Needs internet at runtime | No | Yes |
| Runs on a hosted server | Not practically | **Yes** |

Ingestion (OCR, image understanding, transcription, embeddings, retrieval,
reranking) is local in **both** modes. Only answer generation differs.

---

## 1. Local — both modes

```powershell
cd C:\Users\Anish\Documents\EvidenceAI
copy .env.example .env      # then edit .env
.\start.ps1
```

For Online mode, set one variable in `.env`:

```
OPENAI_API_KEY=sk-your-real-key-here
```

`start.ps1` no longer exits when Ollama is missing — it warns and starts
anyway, so Online mode works on a machine with no Ollama installed.

---

## 2. Hosted Online deployment (Streamlit Community Cloud)

The public user needs **no Ollama, no Python, no Git, no Tesseract, no local
models** — only a browser.

### Steps

1. **Push to GitHub.** Confirm `.env` is not included:
   ```bash
   git check-ignore .env && echo "ignored - safe to push"
   ```
2. Go to <https://share.streamlit.io> and connect the repository.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-your-real-key-here"
   INFERENCE_BACKEND = "online"
   ```
   These are stored by the platform, never in Git. `config.py` bridges
   `st.secrets` into the environment automatically.
5. Deploy. Your public URL will be
   `https://<app-name>.streamlit.app`.

### Files that make this work

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies |
| `packages.txt` | apt packages — Tesseract OCR and `libgl1` for OpenCV |
| `.env.example` | documents every variable; placeholders only |

### Honest constraints of the hosted build

These are real and worth knowing before you rely on a hosted demo:

- **Offline mode will show "not ready" on the host.** There is no Ollama
  there. That is correct behaviour, not a bug — the UI offers *Switch to
  Online*.
- **The index does not travel by default.** `index/` is gitignored, so a
  fresh deployment starts empty and users must ingest through the UI. To
  ship a prepopulated corpus, commit `index/` deliberately — and only if the
  documents in it are safe to make public.
- **Resource limits.** The Community Cloud tier gives ~1 GB RAM. `torch`,
  `faiss` and `faster-whisper` together are close to that ceiling. Document
  and image ingestion are fine; audio transcription may be tight.
- **Evidence leaves the machine in Online mode.** Retrieved evidence is sent
  to the cloud provider to write the answer. For sensitive material, use
  Offline mode locally.

---

## 3. Other hosts

Any container host works. The application is a normal Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT
```

Supply `OPENAI_API_KEY` through the platform's secret manager — never in the
image or the repository. `ONLINE_BASE_URL` lets you point at any
OpenAI-compatible endpoint (Azure OpenAI, OpenRouter, an internal gateway)
without changing code.

---

## 4. Security checklist before pushing

```bash
git check-ignore .env                                  # must print .env
git ls-files | grep -x ".env" && echo "STOP"           # must find nothing
git grep -nE "sk-(proj-)?[A-Za-z0-9_-]{20,}" HEAD      # must find nothing
```

The key is only ever read from configuration and sent in the `Authorization`
header. It is never logged, never rendered in the UI, and is redacted from
API error bodies before they reach a message.
